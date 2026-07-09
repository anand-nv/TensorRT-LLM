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
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <vector>

#include <cuda_fp16.h>

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
    numTokens = numVocabs;
    TLLM_LOG_INFO("TrtLocalTransformer constructor called with numVocabs=%d, maxNumSequences=%d, maxSequenceLen=%d, numMicroBatches=%d, maxBatchSize=%d", 
        numVocabs, maxNumSequences, maxSequenceLen, numMicroBatches, maxBatchSize);
    TLLM_LOG_INFO("TrtLocalTransformer: Creating %d decoder buffers (numTokens=%d). Note: vocabSize=%d is the vocabulary size, not the number of buffers.", 
        numVocabs, numTokens, vocabSize);

    mRuntime->clearContexts();
    auto const contextId = 0;
    mRuntime->addContext(contextId);
    auto& manager = getBufferManager();
    // Cache static engine descriptors at construction — they don't change per step.
    mHiddenStatesType = mRuntime->getEngine().getTensorDataType(kInHiddenStatesTensorName);
    mRandUniformType = mRuntime->getEngine().getTensorDataType(kRandUniformTensorName);
    for (std::int32_t i = 0; i < mRuntime->getEngine().getNbIOTensors(); ++i)
    {
        auto const* const name = mRuntime->getEngine().getIOTensorName(i);
        if (std::strcmp(name, kEosPolicyTensorName) == 0
            && mRuntime->getEngine().getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
        {
            mHasEosPolicyInput = true;
            mEosPolicyType = mRuntime->getEngine().getTensorDataType(kEosPolicyTensorName);
            break;
        }
    }
    inHiddenStates = manager.emptyTensor(MemoryType::kGPU, mHiddenStatesType);
    inHiddenStatesHost = manager.emptyTensor(MemoryType::kCPU, mHiddenStatesType);
    inRandUniform = manager.emptyTensor(MemoryType::kGPU, mRandUniformType);
    inRandUniformHost = manager.emptyTensor(MemoryType::kCPU, mRandUniformType);
    if (mHasEosPolicyInput)
    {
        inEosPolicy = manager.emptyTensor(MemoryType::kGPU, mEosPolicyType);
        inEosPolicyHost = manager.emptyTensor(MemoryType::kCPU, mEosPolicyType);
    }
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

    // Pre-allocate GPU/CPU buffers at max batch size (CFG doubles batch, so *2).
    // Avoids one GPU/CPU malloc per run() call on the hot path; stable addresses also
    // enable CUDA-graph capture of the enqueueV3 + D2H copy below.
    mMaxBatchSize = maxBatchSize;
    auto const maxBufBatch = maxBatchSize * 2;
    mInHiddenStatesBuf  = manager.gpu(ITensor::makeShape({maxBufBatch, hiddenSize}), mHiddenStatesType);
    if (mHasEosPolicyInput)
    {
        mInEosPolicyBuf = manager.gpu(ITensor::makeShape({maxBufBatch}), mEosPolicyType);
        mInEosPolicyHostBuf = manager.cpu(ITensor::makeShape({maxBufBatch}), mEosPolicyType);
    }
    mOutLogitsBuf       = manager.gpu(ITensor::makeShape({maxBufBatch, numTokens}),
                                      nvinfer1::DataType::kINT32);
    mOutLogitsHostBuf   = manager.cpu(ITensor::makeShape({maxBufBatch, numTokens}),
                                      nvinfer1::DataType::kINT32);  // pre-alloc avoids per-step CPU malloc
    mReorderedStatesBuf = manager.gpu(ITensor::makeShape({maxBufBatch, hiddenSize}), mHiddenStatesType);
}

TrtLocalTransformer::~TrtLocalTransformer()
{
}

void TrtLocalTransformer::HandleLogits(
    RequestVector const& contextRequests,
    RequestVector const& generationRequests) {
    // forward the logits to the decoder buffers
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
                auto decoderLogits = ITensor::view(logitsView, ITensor::makeShape({1, 1}));

                auto& decodingOutput = mDecoderBuffers.at(seqSlot)->getJointDecodingOutput();
                auto tokensDest = ITensor::slice(decodingOutput.newTokensSteps, tensor_offset, numFrames);
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
    TLLM_CHECK_WITH_INFO(batchSize > 0, "Local transformer batch size must be positive");
    auto const minProfileShape = mRuntime->getEngine().getProfileShape(
        kInHiddenStatesTensorName, 0, nvinfer1::OptProfileSelector::kMIN);
    auto const maxProfileShape = mRuntime->getEngine().getProfileShape(
        kInHiddenStatesTensorName, 0, nvinfer1::OptProfileSelector::kMAX);
    auto const minProfileBatch = static_cast<SizeType32>(minProfileShape.d[0]);
    auto const maxProfileBatch = static_cast<SizeType32>(maxProfileShape.d[0]);
    auto const effectiveBatchSize = std::max(batchSize, minProfileBatch);
    TLLM_CHECK_WITH_INFO(effectiveBatchSize <= maxProfileBatch,
        "Local transformer batch size %d exceeds max profile batch %d", effectiveBatchSize, maxProfileBatch);

    auto& manager = getBufferManager();
    // Re-use pre-allocated GPU buffers — slice to actual batch size to avoid per-step malloc.
    inHiddenStates = ITensor::slice(mInHiddenStatesBuf, 0, effectiveBatchSize);
    outLogits      = ITensor::slice(mOutLogitsBuf,      0, effectiveBatchSize);

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
            TensorPtr statesView = ITensor::slice(hiddenStates, frameIndex-numFrames, numFrames);
            TensorPtr outStatesView = ITensor::slice(inHiddenStates, batchIndex, numFrames);
            manager.copy(*statesView, *outStatesView);
            batchIndex += numFrames;
        }
    }

    // for generation requests, copy the rest of the runtime buffer to the local transformer buffer
    if (generationRequests.size() > 0)
    {
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
        TensorPtr reorderedStates = ITensor::slice(mReorderedStatesBuf, 0, effectiveBatchSize);
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
    if (effectiveBatchSize > batchSize)
    {
        for (SizeType32 row = batchSize; row < effectiveBatchSize; ++row)
        {
            manager.copy(*ITensor::slice(inHiddenStates, batchSize - 1, 1),
                         *ITensor::slice(inHiddenStates, row, 1));
        }
    }

    sync_check_cuda_error(mRuntime->getStream().get());

    inRandUniformHost = manager.cpu(ITensor::makeShape({numTokens, effectiveBatchSize}), mRandUniformType);
    inRandUniform = manager.gpu(ITensor::makeShape({numTokens, effectiveBatchSize}), mRandUniformType);
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_real_distribution<float> randUniformDist(0.0F, 1.0F);
    auto const randCount = static_cast<std::size_t>(numTokens) * static_cast<std::size_t>(effectiveBatchSize);
    if (mRandUniformType == nvinfer1::DataType::kFLOAT)
    {
        auto* randData = reinterpret_cast<float*>(inRandUniformHost->data());
        for (std::size_t i = 0; i < randCount; ++i)
        {
            randData[i] = randUniformDist(rng);
        }
    }
    else if (mRandUniformType == nvinfer1::DataType::kHALF)
    {
        auto* randData = reinterpret_cast<__half*>(inRandUniformHost->data());
        for (std::size_t i = 0; i < randCount; ++i)
        {
            randData[i] = __float2half(randUniformDist(rng));
        }
    }
    else
    {
        TLLM_THROW("Unsupported local transformer rand_uniform dtype");
    }
    manager.copy(*inRandUniformHost, *inRandUniform);

    if (mHasEosPolicyInput)
    {
        inEosPolicy = ITensor::slice(mInEosPolicyBuf, 0, effectiveBatchSize);
        inEosPolicyHost = ITensor::slice(mInEosPolicyHostBuf, 0, effectiveBatchSize);

        auto setPolicy = [&](SizeType32 idx, float value)
        {
            if (mEosPolicyType == nvinfer1::DataType::kFLOAT)
            {
                reinterpret_cast<float*>(inEosPolicyHost->data())[idx] = value;
            }
            else if (mEosPolicyType == nvinfer1::DataType::kHALF)
            {
                reinterpret_cast<__half*>(inEosPolicyHost->data())[idx] = __float2half(value);
            }
            else
            {
                TLLM_THROW("Unsupported local transformer eos_policy dtype");
            }
        };

        for (SizeType32 row = 0; row < effectiveBatchSize; ++row)
        {
            setPolicy(row, 0.0F);
        }

        auto const eosPolicyForRequest = [&](std::shared_ptr<LlmRequest> const& llmReq, bool isContextRequest)
        {
            static constexpr SizeType32 kMinGeneratedFrames = 4;
            static bool const ignoreFinishedSentenceTracking = []() {
                char const* env = std::getenv("TRT_EOS_IGNORE_FINISHED_SENTENCE_TRACKING");
                return env == nullptr || env[0] == '\0' || std::atoi(env) != 0;
            }();
            SizeType32 const completedSteps = isContextRequest
                ? 0
                : (llmReq->getNumTokens(0) - llmReq->mPromptLen);
            bool const tooEarlyForEos = completedSteps < kMinGeneratedFrames;
            if (tooEarlyForEos)
            {
                return -1.0F;
            }
            if (!mModelConfig.useAttentionPrior())
            {
                return 0.0F;
            }
            // NeMo's default inference sets ignore_finished_sentence_tracking=True.
            // In that mode unfinished_items and finished_items are both empty, so
            // audio EOS is not gated by text-focus tracking after the minimum frame
            // guard. Keep the older prior-gated behavior available with
            // TRT_EOS_IGNORE_FINISHED_SENTENCE_TRACKING=0 for A/B validation.
            if (ignoreFinishedSentenceTracking)
            {
                return 0.0F;
            }
            // Match NeMo's three EOS states:
            //   unfinished text: forbid EOS,
            //   near text end: allow the model to choose EOS naturally,
            //   end counter expired: force EOS.
            if (llmReq->isAttentionPriorFinished())
            {
                return 1.0F;
            }
            return llmReq->isAttentionPriorNearEnd() ? 0.0F : -1.0F;
        };

        auto tracePolicy = [&](std::shared_ptr<LlmRequest> const& llmReq, bool isContextRequest,
                               SizeType32 row, float value)
        {
            char const* tracePath = std::getenv("TRT_EOS_POLICY_TRACE");
            if (tracePath == nullptr || tracePath[0] == '\0')
            {
                return;
            }
            static constexpr SizeType32 kMinGeneratedFrames = 4;
            static bool const ignoreFinishedSentenceTracking = []() {
                char const* env = std::getenv("TRT_EOS_IGNORE_FINISHED_SENTENCE_TRACKING");
                return env == nullptr || env[0] == '\0' || std::atoi(env) != 0;
            }();
            SizeType32 const completedSteps = isContextRequest
                ? 0
                : (llmReq->getNumTokens(0) - llmReq->mPromptLen);
            bool const hasFocus = llmReq->hasAttentionPriorIdx();
            auto const focus = hasFocus ? llmReq->getAttentionPriorIdx(mModelConfig) : -1;
            std::ofstream out(tracePath, std::ios::app);
            out << "eos_policy"
                << "\treq=" << llmReq->mRequestId
                << "\trow=" << row
                << "\tcontext=" << (isContextRequest ? 1 : 0)
                << "\tvalue=" << value
                << "\tsteps=" << completedSteps
                << "\ttoo_early=" << (completedSteps < kMinGeneratedFrames ? 1 : 0)
                << "\tfocus=" << focus
                << "\tenc_len=" << llmReq->getEncoderOutputLen()
                << "\tnear_end=" << (llmReq->isAttentionPriorNearEnd() ? 1 : 0)
                << "\tfinished=" << (llmReq->isAttentionPriorFinished() ? 1 : 0)
                << "\tignore_fst=" << (ignoreFinishedSentenceTracking ? 1 : 0)
                << "\tend_count=" << llmReq->getAttentionPriorEndAttendCount()
                << "\tmax_end=" << llmReq->getMaxEndAttendCount()
                << "\n";
        };

        // After CFG reordering, rows [0, numRequests) are the conditional rows used by
        // IntLT for sampling. Rows [numRequests, 2*numRequests) are unconditional and ignored.
        SizeType32 policyRow = 0;
        for (auto const& llmReq : contextRequests)
        {
            auto const value = eosPolicyForRequest(llmReq, true);
            setPolicy(policyRow, value);
            tracePolicy(llmReq, true, policyRow, value);
            policyRow++;
        }
        for (auto const& llmReq : generationRequests)
        {
            auto const value = eosPolicyForRequest(llmReq, false);
            setPolicy(policyRow, value);
            tracePolicy(llmReq, false, policyRow, value);
            policyRow++;
        }

        manager.copy(*inEosPolicyHost, *inEosPolicy);
    }

    inputMap.clear();
    outputMap.clear();
    // always needs the hidden states as input
    inputMap.insert_or_assign(kInHiddenStatesTensorName, inHiddenStates);
    inputMap.insert_or_assign(kRandUniformTensorName, inRandUniform);
    if (mHasEosPolicyInput)
    {
        inputMap.insert_or_assign(kEosPolicyTensorName, inEosPolicy);
    }

    // puts logits into the same buffer
    outputMap.insert_or_assign(kOutLogitsTensorName, outLogits);
    // Bind any optional non-logits outputs so engines with marked intermediate
    // tensors remain executable. The runtime consumes only logits.
    std::vector<TensorPtr> extraOutputTensors;
    for (std::int32_t tensorIdx = 0; tensorIdx < mRuntime->getEngine().getNbIOTensors(); ++tensorIdx)
    {
        auto const* const outputName = mRuntime->getEngine().getIOTensorName(tensorIdx);
        if (std::strcmp(outputName, kOutLogitsTensorName) == 0
            || mRuntime->getEngine().getTensorIOMode(outputName) != nvinfer1::TensorIOMode::kOUTPUT)
        {
            continue;
        }

        auto outputShape = mRuntime->getEngine().getTensorShape(outputName);
        for (int dimIdx = 0; dimIdx < outputShape.nbDims; ++dimIdx)
        {
            if (outputShape.d[dimIdx] < 0)
            {
                outputShape.d[dimIdx] = effectiveBatchSize;
            }
        }
        TensorPtr extraOutput = manager.gpu(outputShape, mRuntime->getEngine().getTensorDataType(outputName));
        outputMap.insert_or_assign(outputName, extraOutput);
        extraOutputTensors.emplace_back(std::move(extraOutput));
    }
    // provide additional input - previously generated tokens

    auto const contextId = 0;
    // Set input and output tensors on the context before executing
    mRuntime->setInputTensors(contextId, inputMap);
    mRuntime->setOutputTensors(contextId, outputMap);

    // Check actual logits width after setOutputTensors.
    auto const& logitsShape = outLogits->getShape();
    TLLM_CHECK_WITH_INFO(logitsShape.nbDims >= 2 && logitsShape.d[1] > 0,
        "Unexpected local transformer logits shape: %s", ITensor::toString(logitsShape).c_str());
    auto const logitsWidth = static_cast<SizeType32>(logitsShape.d[1]);

    // Use pre-allocated host buffer when width matches; fall back to dynamic alloc otherwise.
    if (logitsWidth == numTokens)
    {
        outLogitsHost = ITensor::slice(mOutLogitsHostBuf, 0, numRequests);
    }
    else
    {
        outLogitsHost = manager.cpu(ITensor::makeShape({numRequests, logitsWidth}), nvinfer1::DataType::kINT32);
    }

    auto outLogitsSlice = ITensor::slice(outLogits, 0, numRequests);

    if (!mRuntime->executeContext(contextId))
        throw std::runtime_error("Executing local transformer engine failed!");
    manager.copy(*outLogitsSlice, *outLogitsHost);
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
    TLLM_LOG_DEBUG("Getting decoder buffers for vocabId %d", vocabId);
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
    
    // Get generated tokens on host. The local transformer emits one token per vocab channel.
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
            
            // Check finish conditions. NeMo's default detector stops when either
            // multinomial or argmax emits EOS on any codebook/frame-stack channel.
            bool shouldFinish = false;
            auto const tokenRow = tokensData + static_cast<size_t>(batchIndex) * rowStride;

            // Suppress EOS for the first min_generated_frames codec frames.
            // completedSteps = getNumTokens(0) - mPromptLen = n_steps * stacking_factor (in frames).
            // Each step adds stacking_factor positions, so completedSteps already counts frames.
            // Do NOT multiply by kStackingFactor again — that would double-count.
            static constexpr SizeType32 kMinGeneratedFrames = 4;
            SizeType32 const completedSteps = isContextRequest
                ? 0
                : (llmReq->getNumTokens(0) - llmReq->mPromptLen);
            bool const tooEarlyForEos = completedSteps < kMinGeneratedFrames;
            bool const eosAllowedByAttentionPrior = !mModelConfig.useAttentionPrior()
                || llmReq->isAttentionPriorNearEnd() || llmReq->isAttentionPriorFinished();

            // Check if EOS token
            auto const endId = llmReq->mEndId.value();
            bool anyCodebookEos = false;
            for (SizeType32 col = 0; col < rowStride; ++col)
            {
                if (tokenRow[col] == endId)
                {
                    anyCodebookEos = true;
                    break;
                }
            }
            if (!tooEarlyForEos && eosAllowedByAttentionPrior && anyCodebookEos)
            {
                shouldFinish = true;
                TLLM_LOG_DEBUG("Request %lu hit EOS token", llmReq->mRequestId);
            }
            else if (!tooEarlyForEos && !eosAllowedByAttentionPrior && anyCodebookEos)
            {
                TLLM_LOG_DEBUG("Request %lu generated EOS before attention prior finished; suppressing finish",
                    llmReq->mRequestId);
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
