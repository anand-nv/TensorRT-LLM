/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <algorithm>

#include "tensorrt_llm/batch_manager/trtLocalTransformer.h"

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/decoderBuffers.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/batch_manager/makeDecodingBatchInputOutput.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/common/tllmException.h"
#include "tensorrt_llm/kernels/decodingCommon.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/utils/runtimeUtils.h"

namespace tensorrt_llm::batch_manager
{

using namespace tensorrt_llm::runtime;

TrtLocalTransformer::TrtLocalTransformer(
    runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, runtime::RawEngine const& rawEngine,
    SizeType32 numVocabs,  // Number of decoder buffers to create (typically 1-8, NOT the vocabulary size)
    std::shared_ptr<nvinfer1::ILogger> logger,
    SizeType32 maxNumSequences,
    SizeType32 maxSequenceLen,
    SizeType32 numMicroBatches,
    SizeType32 maxBatchSize)
    : mModelConfig{modelConfig}
    , mWorldConfig{worldConfig}
    , mDevice{runtime::utils::initDevice(worldConfig)}
    , mRuntime{std::make_shared<TllmRuntime>(rawEngine, logger.get(), false, 1.0f)}
    , hiddenSize{768}  // TODO: change to 768 once switch to use hidden state instead of logits from model
    , vocabSize{1032}  // Actual vocabulary size for the decoder
    , mMaxNumSequences{maxNumSequences}
{
    TLLM_LOG_INFO("TrtLocalTransformer constructor called with numVocabs=%d, maxNumSequences=%d, maxSequenceLen=%d, numMicroBatches=%d, maxBatchSize=%d", 
        numVocabs, maxNumSequences, maxSequenceLen, numMicroBatches, maxBatchSize);
    TLLM_LOG_INFO("TrtLocalTransformer: Creating %d decoder buffers (numTokens=%d). Note: vocabSize=%d is the vocabulary size, not the number of buffers.", 
        numVocabs, numTokens, vocabSize);

    mRuntime->clearContexts();
    auto const contextId = 0;
    numTokens = numVocabs; 
    mstackingFactor = modelConfig.getStackingFactor();
    mRuntime->addContext(contextId);
    auto& manager = getBufferManager();
    auto const statesType = mRuntime->getEngine().getTensorDataType(kInHiddenStatesTensorName);
    inHiddenStates = manager.emptyTensor(MemoryType::kGPU, statesType);
    inHiddenStatesHost = manager.emptyTensor(MemoryType::kCPU, statesType);
    outLogits = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    outLogitsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);

    // create decoder and buffers
    auto decodingMode = executor::DecodingMode::TopKTopP();
    auto decoderType = mModelConfig.getDataType();
    
    // Create and setup the decoder
    mDecoder = std::make_shared<runtime::GptDecoderBatched>(getRuntimeStreamPtr());
    mDecoder->setup(decodingMode, mMaxNumSequences, 1 /*beamWidth*/, decoderType, 
                    mModelConfig, mWorldConfig, vocabSize);
    TLLM_LOG_INFO("Creating local transformer decoder input buffers for %d micro batches", numMicroBatches);
    for (SizeType32 i = 0; i < numMicroBatches; ++i)
    {
        mDecoderInputBuffers.emplace_back(
            maxBatchSize, mModelConfig.getMaxDecodingTokens(), getBufferManager());
    }
    TLLM_LOG_INFO("Creating decoder buffers for %d tokens", numTokens);
    TLLM_CHECK_WITH_INFO(numTokens > 0, 
        "numTokens must be greater than 0, but got %d. This will result in empty mDecoderBuffers vector.", numTokens);
    mDecoderBuffers.reserve(numTokens);
    for (SizeType32 i = 0; i < numTokens; i++) {
        // independent decoder buffer for each token
        TLLM_LOG_INFO("Creating decoder buffer for token %d", i);
        auto decoderState = std::make_shared<runtime::decoder::DecoderState>();
        decoderState->setup(
            mMaxNumSequences, 1 /*beam width*/, 0 /*maxAttentionWindow*/, 
            0 /*sinkTokenLength*/, maxSequenceLen, decoderType,
            mModelConfig, mWorldConfig, getBufferManager()
        );
        TLLM_LOG_DEBUG("Decoder buffer created for token %d", i);
        mDecoderBuffers.push_back(decoderState);
    }
    TLLM_CHECK_WITH_INFO(mDecoderBuffers.size() == numTokens,
        "Failed to create all decoder buffers. Expected %d, but created %zu", numTokens, mDecoderBuffers.size());
    mSlotDecoderBuffers.clear();
    for (SizeType32 i = 0; i < mMaxNumSequences; ++i)
    {
        mSlotDecoderBuffers.emplace_back(std::make_shared<SlotDecoderBuffers>(
            1 /*beam width*/, maxSequenceLen, getBufferManager()));
    }
    mDecodingInputs.resize(numMicroBatches);
    
    // Initialize host buffers for decoder outputs
    auto constexpr nvSizeType = TRTDataType<SizeType32>::value;
    auto constexpr nvFinishedStateType = TRTDataType<tensorrt_llm::kernels::FinishedState::UnderlyingType>::value;
    mSequenceLengthsHost = manager.cpu(ITensor::makeShape({mMaxNumSequences, 1}), nvSizeType);
    mFinishReasonsHost = manager.cpu(ITensor::makeShape({mMaxNumSequences, 1}), nvFinishedStateType);
    mFinishedSumHost = manager.cpu(ITensor::makeShape({mMaxNumSequences}), nvSizeType);
    
    // Initialize to zeros
    manager.setZero(*mSequenceLengthsHost);
    manager.setZero(*mFinishReasonsHost);
    manager.setZero(*mFinishedSumHost);
    
    TLLM_LOG_INFO("TrtLocalTransformer constructor completed successfully. Created %zu decoder buffers, %zu slot decoder buffers", 
        mDecoderBuffers.size(), mSlotDecoderBuffers.size());
}

TrtLocalTransformer::~TrtLocalTransformer()
{
}

void TrtLocalTransformer::HandleLogits(
    RequestVector const& contextRequests,
    RequestVector const& generationRequests) {
    // forward the logits to the decoder buffers
    TLLM_LOG_INFO("numContextRequests: %d", contextRequests.size());
    TLLM_LOG_INFO("numGenerationRequests: %d", generationRequests.size());
    auto numFrames = 1;
    for (auto const& requests : {contextRequests, generationRequests})
    {
        SizeType32 batchIndex{0};
        for (SizeType32 i = 0; i < numTokens; i++) {
            for (auto const& llmReq : requests)
            {
                auto tensor_offset = batchIndex;
                auto const seqSlot = llmReq->mSeqSlots.at(0);


                TensorPtr logitsView = ITensor::slice(outLogits, tensor_offset, numFrames);
                TLLM_LOG_INFO("logitsView shape: %s", ITensor::toString(logitsView->getShape()).c_str());
                auto decoderLogits = ITensor::view(logitsView, ITensor::makeShape({1, 1}));
                TLLM_LOG_INFO("decoderLogits shape: %s", ITensor::toString(decoderLogits->getShape()).c_str());



                auto& decodingOutput = mDecoderBuffers.at(seqSlot)->getJointDecodingOutput();
                auto tokensDest = ITensor::slice(decodingOutput.newTokensSteps, tensor_offset, numFrames);
                TLLM_LOG_INFO("decoderBuffer->newTokensSteps shape: %s", ITensor::toString(tokensDest->getShape()).c_str());
                mRuntime->getBufferManager().copy(*decoderLogits, *tokensDest);
                batchIndex += numFrames;
            }
        }
        batchIndex += numFrames;
    }
}


void TrtLocalTransformer::run(TensorPtr const& hiddenStates,
    RequestVector const& contextRequests,
    std::vector<SizeType32> const& numContextFramesVec,
    RequestVector const& generationRequests,
    SizeType32 microBatchId,
    SizeType32 fusedBufferId
) {
    // create a context for the local transformer engine
    // reshape input hidden states based on the inputs and fill it in using requests
    // check that all requests are actually cfg
    bool allCfg = true;
    bool hasCfg = false;
    for (auto const& req : contextRequests) {
        allCfg &= req->isCfg();
        hasCfg |= req->isCfg();
    }
    for (auto const& req : generationRequests) {
        allCfg &= req->isCfg();
        hasCfg |= req->isCfg();
    }
    if (!allCfg && hasCfg) {
        TLLM_LOG_ERROR("Either all requests should be CFG or none");
    }
    // that specifies whether each request corresponds to dim * 2 or just dim of hidden states
    int cfgMult = allCfg ? 2 : 1;

    // overall batch size (includes CFG doubling if enabled)
    auto const batchSize = (int)(contextRequests.size() + generationRequests.size()) * cfgMult;
    // actual number of requests (without CFG doubling)
    auto const numRequests = (int)(contextRequests.size() + generationRequests.size());

    auto& manager = getBufferManager();
    auto const statesType = mRuntime->getEngine().getTensorDataType(kInHiddenStatesTensorName);
    inHiddenStates = manager.gpu(ITensor::makeShape({batchSize, hiddenSize}), statesType);

    // Output buffer batch dim must match hidden states (CFG doubles batch). Second dim is an upper bound;
    // setOutputTensors reshapes logits to the engine-inferred width (may be numTokens without stacking).
    outLogits = manager.gpu(ITensor::makeShape({batchSize, numTokens * mstackingFactor}), nvinfer1::DataType::kINT32);

    // for context requests, copy the hidden states into the input buffer
    SizeType32 batchIndex{0};
    SizeType32 frameIndex{0};
    for (auto const& llmReq : contextRequests) {

        auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
        //TLLM_CHECK_WITH_INFO(reqBeamWidth == 1, "Beam width must be 1 for local transformer");
        auto const contextFrames = numContextFramesVec.at(batchIndex);
        //TLLM_CHECK_WITH_INFO(!llmReq->isLastContextChunk() || llmReq->getNumDraftTokens() == 0,
        //    "Draft tokens are not supported for local transformer");

        // copy hidden states for conditional (and optionally unconditional) generation
        for (SizeType32 i = 0; i < cfgMult; i++) {
            frameIndex += contextFrames;
            auto const numFrames = 1;
            #ifndef NDEBUG
            auto n_dims = hiddenStates->getShape().nbDims;
            for (int i_ = 0; i_ < n_dims; i_++) {
                TLLM_LOG_INFO("hiddenStates %d: %d", i_, hiddenStates->getShape().d[i_]);
            }
            for (int i_ = 0; i_ < inHiddenStates->getShape().nbDims; i_++) {
                TLLM_LOG_INFO("inHiddenStates %d: %d", i_, inHiddenStates->getShape().d[i_]);
            }
            TLLM_LOG_DEBUG("Copying hidden states for request %d %d", batchIndex, i);
            TLLM_LOG_DEBUG("frameIndex1: %d %d", frameIndex, i);
            TLLM_LOG_DEBUG("numFrames:1 %d %d", numFrames, i);
            TLLM_LOG_DEBUG("batchIndex1: %d %d", batchIndex, i);
            #endif
            TensorPtr statesView = ITensor::slice(hiddenStates, frameIndex-numFrames, numFrames);
            TensorPtr outStatesView = ITensor::slice(inHiddenStates, batchIndex, numFrames);
            
            #ifndef NDEBUG
            TLLM_LOG_DEBUG("Checking if cpp only works1.");
            for (int i_ = 0; i_ < statesView->getShape().nbDims; i_++) {
                TLLM_LOG_DEBUG("statesView %d: %d", i_, statesView->getShape().d[i_]);
            }
            for (int i_ = 0; i_ < outStatesView->getShape().nbDims; i_++) {
                TLLM_LOG_DEBUG("outStatesView %d: %d", i_, outStatesView->getShape().d[i_]);
            }
            #endif
            manager.copy(*statesView, *outStatesView);
            batchIndex += numFrames;
        }
    }

    // for generation requests, copy the rest of the runtime buffer to the local transformer buffer
    #ifndef NDEBUG
        TLLM_LOG_INFO("Copying hidden states to local transformer buffer");
    #endif
    if (generationRequests.size() > 0) {
        TensorPtr genStatesView = ITensor::slice(hiddenStates, frameIndex, generationRequests.size() * cfgMult);
        TensorPtr outGenStatesView = ITensor::slice(inHiddenStates, batchIndex, generationRequests.size() * cfgMult);
        manager.copy(*genStatesView, *outGenStatesView);
        batchIndex += generationRequests.size() * cfgMult;
        frameIndex += generationRequests.size() * cfgMult;
    }
    // The NeMo/TRT engine (IntLT.forward) splits the input at mid-batch:
    //   cond_logits = output[:B], uncond_logits = output[B:]
    // So it expects NON-INTERLEAVED layout:
    //   rows 0..N-1:   conditional   (req0, req1, ..., req(N-1))
    //   rows N..2N-1:  unconditional (req0, req1, ..., req(N-1))
    //
    // runtimeBuffers fills INTERLEAVED per request:
    //   rows 0,1 = req0_cond, req0_uncond; rows 2,3 = req1_cond, req1_uncond …
    //
    // Reorder to non-interleaved so the engine computes correct CFG pairs.
    if (cfgMult == 2) {
        TensorPtr reorderedStates = manager.gpu(ITensor::makeShape({batchSize, hiddenSize}), statesType);
        for (SizeType32 i = 0; i < numRequests; i++) {
            // Even rows (2i)   → conditional half (rows 0..N-1)
            manager.copy(*ITensor::slice(inHiddenStates, 2 * i, 1),
                         *ITensor::slice(reorderedStates, i, 1));
            // Odd rows (2i+1) → unconditional half (rows N..2N-1)
            manager.copy(*ITensor::slice(inHiddenStates, 2 * i + 1, 1),
                         *ITensor::slice(reorderedStates, numRequests + i, 1));
        }
        inHiddenStates = reorderedStates;
    }

    sync_check_cuda_error(mRuntime->getStream().get());

    // Dump inHiddenStates to /tmp for comparison with NeMo dec_output.
    // Only save the first 3 calls (context + first 2 gen steps) to avoid filling disk.
    // Format: raw float16 (half) elements, shape (batchSize, hiddenSize).
    //{
    //    static int sDumpCount = 0;
    //    if (sDumpCount < 3) {
    //        auto hostDump = manager.cpu(inHiddenStates->getShape(), statesType);
    //        manager.copy(*inHiddenStates, *hostDump);
    //        mRuntime->getStream().synchronize();
    //        // element size in bytes: fp16=2, fp32=4
    //        size_t const elemBytes = (statesType == nvinfer1::DataType::kFLOAT) ? 4u : 2u;
    //        char path[256];
    //        snprintf(path, sizeof(path),
    //            "/tmp/lt_hidden_states_%d_nreq%d_bs%d_h%d_fp%zu.bin",
    //            sDumpCount, numRequests, batchSize, hiddenSize, elemBytes * 8);
    //        FILE* f = fopen(path, "wb");
    //        if (f) {
    //            fwrite(hostDump->data(), hostDump->getSize() * elemBytes, 1, f);
    //            fclose(f);
    //            printf("Dumped inHiddenStates[%d] shape=(%d,%d) dtype=fp%zu to %s\n",
    //                sDumpCount, batchSize, hiddenSize, elemBytes * 8, path);
    //        }
    //        sDumpCount++;
    //    }
    //}

    inputMap.clear();
    outputMap.clear();
    // always needs the hidden states as input
    inputMap.insert_or_assign(kInHiddenStatesTensorName, inHiddenStates);
    // puts logits into the same buffer
    outputMap.insert_or_assign(kOutLogitsTensorName, outLogits);
    // provide additional input - previously generated tokens

    #ifndef NDEBUG
        TLLM_LOG_INFO("Cleared input/output map");
        // Call TrT engine.
        TLLM_LOG_DEBUG("Running Enqueue");
    #endif
    auto const contextId = 0;
    // Set input and output tensors on the context before executing
    mRuntime->setInputTensors(contextId, inputMap);
    mRuntime->setOutputTensors(contextId, outputMap);

    // Host logits must match the reshaped GPU tensor (engine last dim can differ from numTokens*mstackingFactor).
    auto const& logitsShape = outLogits->getShape();
    TLLM_CHECK_WITH_INFO(logitsShape.nbDims >= 2 && logitsShape.d[1] > 0,
        "Unexpected local transformer logits shape: %s", ITensor::toString(logitsShape).c_str());
    outLogitsHost = manager.cpu(ITensor::makeShape({numRequests, logitsShape.d[1]}), nvinfer1::DataType::kINT32);

    auto enqueueSuccessful = mRuntime->executeContext(contextId);
    if (!enqueueSuccessful)
    {
        throw std::runtime_error("Executing local transformer engine failed!");
    }
    #ifndef NDEBUG
        TLLM_LOG_INFO("Enqueue successful");
        TLLM_LOG_DEBUG("Finished Enqueue");
        sync_check_cuda_error(mRuntime->getStream().get());
        TLLM_LOG_DEBUG("Ran Decoder Syncing cuda error");
        TLLM_LOG_DEBUG("local transformer engine executed");
        
        
        // Copy outLogits to outLogitsHost: first numRequests rows (conditional branch), full inferred width.
        TLLM_LOG_DEBUG("outLogitsHost shape: %s", ITensor::toString(outLogitsHost->getShape()).c_str());
        TLLM_LOG_DEBUG("outLogits shape: %s", ITensor::toString(outLogits->getShape()).c_str());
        TLLM_LOG_DEBUG("Copying outLogits to outLogitsHost (copying first %d rows out of %d, cfgMult=%d)", numRequests, batchSize, cfgMult);
    #endif
    auto outLogitsSlice = ITensor::slice(outLogits, 0, numRequests);
    #ifndef NDEBUG
    TLLM_LOG_DEBUG("outLogitsSlice shape: %s", ITensor::toString(outLogitsSlice->getShape()).c_str());
    TLLM_LOG_DEBUG("Copying outLogits to decoder");
    #endif
    manager.copy(*outLogitsSlice, *outLogitsHost);

    #ifndef NDEBUG
        auto size_tokens = outLogitsHost->getSize();
        auto data_tokens = static_cast<int32_t*>(outLogitsHost->data());
        printf("outLogitsHost data: ");
        for (size_t i = 0; i < size_tokens; i++) {
            TLLM_LOG_DEBUG("outLogits data: %d", data_tokens[i]);
            printf("%d ", data_tokens[i]);
        }
        printf("\n");
        TLLM_LOG_INFO("outLogitsHost shape: %s", ITensor::toString(outLogitsHost->getShape()).c_str());
    #endif
}

TrtLocalTransformer::TensorPtr TrtLocalTransformer::getOutLogitsHost()
{
    return outLogitsHost;
}

std::shared_ptr<runtime::GptDecoderBatched>& TrtLocalTransformer::getDecoder()
{
    return mDecoder;
}

DecoderInputBuffers& TrtLocalTransformer::getDecoderInputBuffers(SizeType32 microBatchId)
{
    return mDecoderInputBuffers.at(microBatchId);
}

std::shared_ptr<runtime::decoder::DecoderState>& TrtLocalTransformer::getDecoderBuffers(SizeType32 vocabId)
{
    TLLM_LOG_INFO("Getting decoder buffers for vocabId %d", vocabId);
    TLLM_CHECK_WITH_INFO(!mDecoderBuffers.empty(), 
        "mDecoderBuffers is empty! This means the constructor did not properly initialize decoder buffers. numTokens=%d", numTokens);
    TLLM_CHECK_WITH_INFO(vocabId < static_cast<SizeType32>(mDecoderBuffers.size()),
        "vocabId %d is out of range for mDecoderBuffers.size()=%zu", vocabId, mDecoderBuffers.size());
    return mDecoderBuffers.at(vocabId);
}

std::shared_ptr<SlotDecoderBuffers>& TrtLocalTransformer::getSlotDecoderBuffers(SizeType32 seqSlot)
{
    return mSlotDecoderBuffers.at(seqSlot);
}

runtime::BufferManager const& TrtLocalTransformer::getBufferManager() const
{
    return mRuntime->getBufferManager();
}

runtime::BufferManager::CudaStreamPtr TrtLocalTransformer::getRuntimeStreamPtr() const
{
    return mRuntime->getStreamPtr();
}

runtime::CudaStream const& TrtLocalTransformer::getRuntimeStream() const
{
    return mRuntime->getStream();
}

SizeType32 TrtLocalTransformer::getVocabSize()
{
    return vocabSize;
}

SizeType32 TrtLocalTransformer::getNumTokens()
{
    return numTokens;
}

void TrtLocalTransformer::clearDecoderBuffers()
{
    mDecoderBuffers.clear();
}

void TrtLocalTransformer::updateDecoderStateAfterGeneration(
    RequestVector const& contextRequests,
    RequestVector const& generationRequests,
    TensorPtr const& generatedTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    
    // Get pointers to our persistent host buffers
    auto seqLengthsData = reinterpret_cast<SizeType32*>(mSequenceLengthsHost->data());
    auto finishData = reinterpret_cast<tensorrt_llm::kernels::FinishedState::UnderlyingType*>(mFinishReasonsHost->data());
    auto finishedSumData = reinterpret_cast<SizeType32*>(mFinishedSumHost->data());
    
    // Get generated tokens on host (should already be on host). Row width is numVocabs * stackingFactor
    // (see TrtLocalTransformer::run outLogitsHost shape), not numTokens alone.
    auto tokensData = reinterpret_cast<SizeType32*>(generatedTokens->data());
    auto const rowStride = static_cast<SizeType32>(generatedTokens->getShape().d[1]);
    TLLM_CHECK_WITH_INFO(rowStride > 0, "generatedTokens row width must be positive");

    // Process both context and generation requests
    SizeType32 batchIndex{0};
    for (auto const& requests : {contextRequests, generationRequests})
    {
        for (auto const& llmReq : requests)
        {
            auto const seqSlot = llmReq->mSeqSlots.at(0);
            auto const reqBeamWidth = llmReq->mSamplingConfig.beamWidth;
            TLLM_CHECK_WITH_INFO(reqBeamWidth == 1, "Beam width must be 1 for local transformer");
            
            // Check if this is a context request (first generation step)
            bool isContextRequest = std::find(contextRequests.begin(), contextRequests.end(), llmReq) 
                                   != contextRequests.end();
            
            // Calculate current sequence length
            SizeType32 currentLength;
            if (isContextRequest)
            {
                currentLength = llmReq->mPromptLen + 1;  // First token after prompt
            }
            else
            {
                currentLength = llmReq->getNumTokens(0) + 1;  // Increment by 1
            }
            
            // Update sequence length in host buffer
            seqLengthsData[seqSlot] = currentLength;
            TLLM_LOG_DEBUG("Updated sequence length for slot %d to %d", seqSlot, currentLength);
            
            // Check finish conditions (EOS on first codebook column; extend if EOS is on another book)
            bool shouldFinish = false;
            SizeType32 const generatedToken = tokensData[static_cast<size_t>(batchIndex) * rowStride];

            // Suppress EOS for the first min_generated_frames codec frames.
            // completedSteps = getNumTokens(0) - mPromptLen = n_steps * stacking_factor (in frames).
            // Each step adds stacking_factor positions, so completedSteps already counts frames.
            // Do NOT multiply by kStackingFactor again — that would double-count.
            static constexpr SizeType32 kMinGeneratedFrames = 4;
            SizeType32 const completedSteps = isContextRequest
                ? 0
                : (llmReq->getNumTokens(0) - llmReq->mPromptLen);
            bool const tooEarlyForEos = completedSteps < kMinGeneratedFrames;

            // Check if EOS token
            auto const endId = llmReq->mEndId.value();
            if (!tooEarlyForEos && generatedToken == endId)
            {
                shouldFinish = true;
                TLLM_LOG_DEBUG("Request %lu hit EOS token", llmReq->mRequestId);
            }
            
            // Check if max sequence length reached
            auto const maxSeqLen = llmReq->mPromptLen + llmReq->mMaxNewTokens;
            if (currentLength >= maxSeqLen)
            {
                shouldFinish = true;
                TLLM_LOG_DEBUG("Request %lu hit max length %d", llmReq->mRequestId, maxSeqLen);
            }
            
            // Update finish reason and sum
            if (shouldFinish)
            {
                finishData[seqSlot] = tensorrt_llm::kernels::FinishedState::finished().toUnderlying();
                finishedSumData[seqSlot] = 1;
            }
            else
            {
                finishData[seqSlot] = tensorrt_llm::kernels::FinishedState::empty().toUnderlying();
                finishedSumData[seqSlot] = 0;
            }
            
            batchIndex++;
        }
    }
    
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

TrtLocalTransformer::TensorPtr TrtLocalTransformer::getSequenceLengthsHost() const
{
    return mSequenceLengthsHost;
}

TrtLocalTransformer::TensorPtr TrtLocalTransformer::getFinishReasonsHost() const
{
    return mFinishReasonsHost;
}

TrtLocalTransformer::TensorPtr TrtLocalTransformer::getFinishedSumHost() const
{
    return mFinishedSumHost;
}

}// namespace tensorrt_llm::batch_manager

