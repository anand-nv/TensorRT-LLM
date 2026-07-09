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

#include "tensorrt_llm/batch_manager/runtimeBuffers.h"

#include "tensorrt_llm/batch_manager/encoderBuffers.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/loraBuffers.h"
#include "tensorrt_llm/batch_manager/medusaBuffers.h"
#include "tensorrt_llm/batch_manager/promptTuningBuffers.h"
#include "tensorrt_llm/batch_manager/rnnStateBuffers.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
#include "tensorrt_llm/batch_manager/transformerBuffers.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/nvtxUtils.h"
#include "tensorrt_llm/common/stlUtils.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/decoderState.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/runtimeKernels.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"

#include <cstdlib>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <numeric>
#include <vector>

using namespace tensorrt_llm::runtime;

namespace tensorrt_llm::batch_manager
{

RuntimeBuffers::RuntimeBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
    TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    executor::DecodingConfig const& decodingConfig, bool gatherGenerationLogits, std::optional<SizeType32> maxNumTokens,
    std::optional<std::vector<executor::AdditionalModelOutput>> const& additionalModelOutputs,
    bool promptTableOffloadingParam)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    promptTableOffloading = promptTableOffloadingParam;

    create(maxBatchSize, maxBeamWidth, maxAttentionWindowVec, maxAttentionWindow, sinkTokenLen, runtime, modelConfig,
        worldConfig, decodingConfig, gatherGenerationLogits, additionalModelOutputs);

    // pre-allocate
    setMaxBufferSizes(maxBatchSize, maxBeamWidth, modelConfig, maxNumTokens);
    reshape(runtime, modelConfig, worldConfig, gatherGenerationLogits);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

RuntimeBuffers::~RuntimeBuffers() = default;

void RuntimeBuffers::create(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
    TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    executor::DecodingConfig const& decodingConfig, bool gatherGenerationLogits,
    std::optional<std::vector<executor::AdditionalModelOutput>> const& additionalModelOutputs,
    TensorPtr priorAttentionPriorScores)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    useAttentionPrior = modelConfig.useAttentionPrior();
    useContextEmbeddings = modelConfig.useContextEmbeddings();
    attentionPriorLookahead = modelConfig.getAttentionPriorLookahead();

    auto const& manager = runtime.getBufferManager();
    auto const& engine = runtime.getEngine();

    if (modelConfig.isTransformerBased())
    {
        transformerBuffers = std::make_unique<TransformerBuffers>(maxBatchSize, maxBeamWidth, maxAttentionWindowVec,
            maxAttentionWindow, sinkTokenLen, runtime, modelConfig, worldConfig);
    }
    if (modelConfig.isRnnBased())
    {
        rnnStateBuffers = std::make_unique<RnnStateBuffers>(maxBatchSize, runtime);
    }

    auto constexpr nvTokenIdType = TRTDataType<TokenIdType>::value;
    inputsIds = manager.emptyTensor(MemoryType::kGPU, nvTokenIdType);

    mropeRotaryCosSin = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
    mropePositionDeltas = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const logitsType = engine.getTensorDataType(batch_manager::RuntimeBuffers::kLogitsTensorName);
        logits = manager.emptyTensor(MemoryType::kGPU, logitsType);
    }

    // TODO: check which tensors can be allocated as pinned for max size
    requestTypes = manager.emptyTensor(MemoryType::kCPU, TRTDataType<runtime::RequestType>::value);

    contextLengthsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    contextLengthsDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    sequenceLengthsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    sequenceLengthsDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    lastTokenIdsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);
    lastTokenIdsDevice = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    logitsIdsHost = manager.emptyTensor(MemoryType::kCPU, nvinfer1::DataType::kINT32);

    inputsIds = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);

    if (useAttentionPrior)
    {
        // probs in attention kernel are in full precision
        attentionPriorScores = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kFLOAT);
        attentionPriorFocus = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kINT32);
    }

    if (useContextEmbeddings)
    {
        auto const featsType = engine.getTensorDataType(kDecoderContextFeaturesTensorName);
        decoderContextFeatures = manager.emptyTensor(MemoryType::kGPU, featsType);
        decoderContextFeaturesMask = manager.emptyTensor(MemoryType::kGPU, nvinfer1::DataType::kBOOL);
    }

    if (worldConfig.isPipelineParallel())
    {
        hiddenStates = manager.emptyTensor(MemoryType::kGPU, modelConfig.getDataType());
    }

    auto const maxBatchSizeShape = ITensor::makeShape({maxBatchSize});
    seqSlots = tensorrt_llm::runtime::BufferManager::pinnedPool(maxBatchSizeShape, nvinfer1::DataType::kINT32);
    seqSlotsDevice = manager.gpu(maxBatchSizeShape, nvinfer1::DataType::kINT32);

    cacheIndirDecoderIOBatchedCopySrcOffsets
        = tensorrt_llm::runtime::BufferManager::pinnedPool(maxBatchSizeShape, nvinfer1::DataType::kINT64);
    cacheIndirDecoderIOBatchedCopyDstOffsets
        = tensorrt_llm::runtime::BufferManager::pinnedPool(maxBatchSizeShape, nvinfer1::DataType::kINT64);
    cacheIndirDecoderIOBatchedCopySizes
        = tensorrt_llm::runtime::BufferManager::pinnedPool(maxBatchSizeShape, nvinfer1::DataType::kINT64);
    mCacheIndirDecoderIOBatchedCopySrcOffsetsSliceDevice = manager.gpu(maxBatchSizeShape, nvinfer1::DataType::kINT64);
    mCacheIndirDecoderIOBatchedCopyDstOffsetsSliceDevice = manager.gpu(maxBatchSizeShape, nvinfer1::DataType::kINT64);
    mCacheIndirDecoderIOBatchedCopyCopySizesDevice = manager.gpu(maxBatchSizeShape, nvinfer1::DataType::kINT64);

    // Pre-allocate buffer for saving generation logits for model w/o draft tokens
    if (gatherGenerationLogits
        && (modelConfig.getSpeculativeDecodingMode().isDraftTokensExternal()
            || modelConfig.getSpeculativeDecodingMode().isNone())
        && worldConfig.isLastPipelineParallelRank())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());
        auto const hiddenSize = modelConfig.getHiddenSize();
        auto const logitsType = engine.getTensorDataType(batch_manager::RuntimeBuffers::kLogitsTensorName);

        generationLogitsCache.transposedLogits = manager.gpu(
            ITensor::makeShape({maxBeamWidth, GenerationLogitsCache::kCACHE_LENGTH, hiddenSize}), logitsType);
        generationLogitsCache.logits = manager.gpu(
            ITensor::makeShape({GenerationLogitsCache::kCACHE_LENGTH, maxBatchSize * maxBeamWidth, hiddenSize}),
            logitsType);

        generationLogitsCache.fragmentPointerDevice
            = manager.gpu(ITensor::makeShape({GenerationLogitsCache::kCACHE_LENGTH}), nvinfer1::DataType::kINT64);
        generationLogitsCache.fragmentPointerHost = tensorrt_llm::runtime::BufferManager::pinnedPool(
            ITensor::makeShape({maxBatchSize, GenerationLogitsCache::kCACHE_LENGTH}), nvinfer1::DataType::kINT64);
    }

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers = std::make_unique<EncoderBuffers>();
        encoderBuffers->create(maxBatchSize, modelConfig, runtime);
    }

    if (modelConfig.usePromptTuning())
    {
        promptTuningBuffers = std::make_unique<PromptTuningBuffers>(
            maxBatchSize, manager, modelConfig, worldConfig, promptTableOffloading);
    }

    if (modelConfig.useLoraPlugin())
    {
        loraBuffers = std::make_unique<LoraBuffers>(maxBatchSize, maxBeamWidth, runtime, modelConfig, worldConfig);
    }

    if (modelConfig.getSpeculativeDecodingMode().isMedusa())
    {
        mMedusaBuffers = std::make_unique<MedusaBuffers>(
            maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig, decodingConfig, runtime);
    }
    else if (modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
    {
        mLookaheadBuffers = std::make_unique<runtime::LookaheadRuntimeBuffers>(
            maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig, decodingConfig, runtime);
    }
    else if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
    {
        mExplicitDraftTokensBuffers = std::make_unique<runtime::ExplicitDraftTokensBuffers>(
            maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig);
    }
    else if (modelConfig.getSpeculativeDecodingMode().isEagle())
    {
        mEagleBuffers = std::make_unique<runtime::EagleBuffers>(
            maxBatchSize, maxBeamWidth, manager, modelConfig, worldConfig, decodingConfig);
    }

    if (modelConfig.useLanguageAdapter())
    {
        languageAdapterRoutings = manager.emptyTensor(MemoryType::kGPU, TRTDataType<SizeType32>::value);
    }

    for (auto const& output : additionalModelOutputs.value_or(std::vector<executor::AdditionalModelOutput>{}))
    {
        auto const& engine = runtime.getEngine();
        auto const dataType = engine.getTensorDataType(output.name.c_str());
        mAdditionalOutputTensors.emplace(output.name, manager.emptyTensor(runtime::MemoryType::kGPU, dataType));
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::setMaxBufferSizes(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
    runtime::ModelConfig const& modelConfig, std::optional<SizeType32> maxNumRuntimeTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    // `maxNumSequences` is reached when all requests are in generation
    numContextRequests = 0;
    numGenRequests = maxBatchSize;
    numGenSequences = maxBatchSize * maxBeamWidth;

    auto const maxDraftTokens = modelConfig.getMaxDecodingDraftTokens();
    // Draft-Tokens and Beam-Search are mutually exclusive
    numLogits = maxBatchSize * std::max(1 + maxDraftTokens, maxBeamWidth);
    auto const maxNumModelTokens = modelConfig.getMaxNumTokens();
    auto const maxNumContextTokens = maxBatchSize * modelConfig.getMaxInputLen();
    auto const maxNumGenTokens = numLogits;
    // For pre-allocation
    numContextTokens = 0; // Set in `setBufferSizes` rather than here for `computeContextLogits`
    numGenTokens
        = maxNumRuntimeTokens.value_or(maxNumModelTokens.value_or(std::max(maxNumContextTokens, maxNumGenTokens)));

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->setMaxBufferSizes(maxBatchSize, modelConfig);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::setBufferSizes(RequestVector const& contextRequests, RequestVector const& genRequests)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersSetBufferSizes);

    // set context sizes
    numContextRequests = 0;
    for (auto const& llmReq : contextRequests)
    {
        numContextRequests += llmReq->getNumSequences();
    }
    auto numContextLogits = numContextRequests;
    numContextTokens = 0;
    maxContextLength = 0;
    for (auto const& llmReq : contextRequests)
    {
        auto const draftLength = llmReq->isLastContextChunk() ? llmReq->getNumDraftTokens() : 0;
        numContextLogits += draftLength * llmReq->getNumSequences();

        auto const contextChunkSize = llmReq->getContextChunkSize();
        numContextTokens += (contextChunkSize + draftLength) * llmReq->getNumSequences();
        if (maxContextLength < llmReq->mPromptLen)
        {
            maxContextLength = llmReq->mPromptLen;
        }
    }

    // set generation sizes
    numGenRequests = 0;
    for (auto const& llmReq : genRequests)
    {
        numGenRequests += llmReq->getNumSequences();
    }
    numGenSequences = 0;
    numGenTokens = 0;
    for (auto const& llmReq : genRequests)
    {
        auto const reqBeamWidth = llmReq->getBeamWidthByIter();
        numGenSequences += reqBeamWidth * llmReq->getNumSequences();
        auto const draftLen = llmReq->getNumDraftTokens();
        numGenTokens += (draftLen + reqBeamWidth) * llmReq->getNumSequences();
    }

    numLogits = numContextLogits + numGenTokens;

    if (encoderBuffers)
    {
        encoderBuffers->setBufferSizes(contextRequests, genRequests);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::reshape(TllmRuntime const& runtime, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    bool gatherGenerationLogits)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersReshape);

    if (worldConfig.isLastPipelineParallelRank())
    {
        auto const vocabSizePadded = modelConfig.getVocabSizePadded(worldConfig.getSize());

        // Use vocabSizePadded for capacity so the TensorView created in fillIOMaps
        // has enough room for whatever the engine's shape inference returns.
        // When the engine is built without lm_head (T5-TTS), the engine infers
        // {N, hiddenSize} and the extra capacity is unused.  When the engine is
        // built with lm_head, the full {N, vocabSizePadded} is needed.
        if (modelConfig.computeContextLogits() && (numContextRequests > 0))
        {
            auto const& engine = runtime.getEngine();
            auto const& manager = runtime.getBufferManager();
            auto const logitsType = engine.getTensorDataType(kLogitsTensorName);
            logits = manager.gpu(ITensor::makeShape({numContextTokens + numGenSequences, vocabSizePadded}), logitsType);
        }
        else if (gatherGenerationLogits && modelConfig.getSpeculativeDecodingMode().isNone())
        {
            logits = ITensor::slice(generationLogitsCache.logits, generationLogitsCache.offset, 1);
            generationLogitsCache.offset = (generationLogitsCache.offset + 1) % GenerationLogitsCache::kCACHE_LENGTH;
            logits->squeeze(0);
        }
        else
        {
            logits->reshape(ITensor::makeShape({numLogits, vocabSizePadded}));
        }
    }

    auto const numSequences = getNumSequences();
    auto const numSequencesShape = ITensor::makeShape({numSequences});
    requestTypes->reshape(numSequencesShape);
    contextLengthsHost->reshape(numSequencesShape);
    contextLengthsDevice->reshape(numSequencesShape);
    sequenceLengthsHost->reshape(numSequencesShape);
    sequenceLengthsDevice->reshape(numSequencesShape);

    auto const numLogitsShape = ITensor::makeShape({numLogits});
    lastTokenIdsHost->reshape(numLogitsShape);
    lastTokenIdsDevice->reshape(numLogitsShape);
    logitsIdsHost->reshape(numLogitsShape);

    if (transformerBuffers)
    {
        transformerBuffers->reshape(numSequences, numContextTokens + numGenTokens);
    }

    if (rnnStateBuffers)
    {
        rnnStateBuffers->reshape(numSequences);
    }

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->reshape();
    }

    if (modelConfig.useLoraPlugin())
    {
        loraBuffers->reshape(numSequences);
    }

    if (mMedusaBuffers)
    {
        mMedusaBuffers->reshape(
            numContextRequests, numGenRequests, modelConfig.getSpeculativeDecodingModulePtr()->getMaxDecodingTokens());
    }

    if (mLookaheadBuffers && modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
    {
        mLookaheadBuffers->reshape(
            numContextRequests, numGenRequests, modelConfig.getSpeculativeDecodingModulePtr()->getMaxDecodingTokens());
    }

    if (mExplicitDraftTokensBuffers)
    {
        mExplicitDraftTokensBuffers->reshape(numContextRequests, numGenRequests, modelConfig);
    }

    if (mEagleBuffers)
    {
        mEagleBuffers->reshape(numContextRequests, numGenRequests, modelConfig);
    }

    auto const numRequests = getNumRequests();
    auto const numRequestsShape = ITensor::makeShape({numRequests});
    seqSlots->reshape(numRequestsShape);
    seqSlotsDevice->reshape(numRequestsShape);

    auto const numTokens = getNumTokens();
    inputsIds->reshape(ITensor::makeShape({numTokens * modelConfig.getNumVocabs()}));

    if (modelConfig.useMrope())
    {
        auto const mropeRotaryCosSinSize = modelConfig.getMaxPositionEmbeddings() * modelConfig.getRotaryEmbeddingDim();
        mropeRotaryCosSin->reshape(ITensor::makeShape({numSequences, mropeRotaryCosSinSize}));
        mropePositionDeltas->reshape(ITensor::makeShape({numSequences, 1}));
    }

    if (worldConfig.isPipelineParallel())
    {
        auto const hiddenSize = (!modelConfig.getPpReduceScatter() || worldConfig.isFirstPipelineParallelRank())
            ? modelConfig.getHiddenSize() * worldConfig.getTensorParallelism()
            : modelConfig.getHiddenSize();

        auto const hiddenStatesShape = ITensor::makeShape({numTokens, hiddenSize});
        hiddenStates->reshape(hiddenStatesShape);
    }

    if (modelConfig.useLanguageAdapter())
    {
        languageAdapterRoutings->reshape(ITensor::makeShape({numTokens, 1}));
    }

    for (auto const& outputTensor : mAdditionalOutputTensors)
    {
        auto const& [name, tensor] = outputTensor;
        auto const& engine = runtime.getEngine();
        auto shape = engine.getTensorShape(name.c_str());
        TLLM_CHECK_WITH_INFO(
            shape.d[0] == -1, "First dimension of additional output tensor '%s' must be dynamic", name.c_str());
        shape.d[0] = numTokens;
        tensor->reshape(shape);
    }

    if (useAttentionPrior)
    {
        attentionPriorScores->reshape(ITensor::makeShape({attentionPriorLookahead * getNumSequences()}));
        attentionPriorFocus->reshape(ITensor::makeShape({getNumSequences()}));
    }

    if (useContextEmbeddings)
    {
        decoderContextFeatures->reshape(ITensor::makeShape({numTokens, modelConfig.getHiddenSize()}));
        decoderContextFeaturesMask->reshape(ITensor::makeShape({numTokens}));
        runtime.getBufferManager().setMem(*decoderContextFeaturesMask, 0);
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareBuffersForCudaGraph(SizeType32 maxSequenceLength)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(prepareBuffersForCudaGraph);

    TLLM_CHECK(numContextRequests == 0);

    if (transformerBuffers)
    {
        // Set pastKeyValueLength for graph capturing. This way we will capture graph with
        // maxKvCacheLengthRounded rounded to the next kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE.
        // MMHA will launch excessive amount of blocks and some of them will exit early during the actual launch.
        // We can reuse the same graph for the next kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE iterations.

        // make sure the size does not overflow the max allowed pastKvCacheLength
        auto const pastKvCacheLength = std::min(maxSequenceLength - 1, maxKvCacheLengthRounded);

        auto* pastKeyValueLengthsPtr = bufferCast<SizeType32>(*transformerBuffers->pastKeyValueLengths);
        std::fill_n(pastKeyValueLengthsPtr, getNumSequences(), pastKvCacheLength);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::setFromInputs(RequestVector const& contextRequests, RequestVector const& genRequests,
    SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, runtime::decoder::DecoderState const& decoderState,
    kv_cache_manager::BaseKVCacheManager* kvCacheManagerPtr,
    kv_cache_manager::BaseKVCacheManager* crossKvCacheManagerPtr,
    rnn_state_manager::RnnStateManager* rnnStateManagerPtr, PeftTable const& peftTable,
    runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
    runtime::WorldConfig const& worldConfig, bool trtOverlap, OptionalRef<runtime::ITensor const> newOutputTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersSetFromInputs);

    auto const& manager = runtime.getBufferManager();
    auto const& stream = runtime.getStream();

    // Fill requestTypes
    {
        auto* hostRequestTypes = bufferCast<runtime::RequestType>(*requestTypes);
        std::fill_n(hostRequestTypes, numContextRequests, runtime::RequestType::kCONTEXT);
        std::fill_n(hostRequestTypes + numContextRequests, numGenSequences, runtime::RequestType::kGENERATION);
    }

    SizeType32 totalInputSize = 0;
    std::vector<TokenIdType> inputHost;
    std::vector<SizeType32> positionIdsHost;
    std::vector<SizeType32> positionIdsHostRow2;
    std::vector<SizeType32> mropePositionDeltasHost;
    std::vector<SizeType32> languageAdapterRoutingsHost;

    auto* contextLengthsHostPtr = bufferCast<SizeType32>(*contextLengthsHost);
    auto* sequenceLengthsHostPtr = bufferCast<SizeType32>(*sequenceLengthsHost);
    auto* pastKeyValueLengthsPtr
        = transformerBuffers ? bufferCast<SizeType32>(*transformerBuffers->pastKeyValueLengths) : nullptr;
    SizeType32 totalNumLogits{0};
    auto* logitsIdsHostPtr = bufferCast<SizeType32>(*logitsIdsHost);
    bool const isChatGlm = modelConfig.getModelVariant() == ModelConfig::ModelVariant::kChatGlm;
    bool const isGlm = modelConfig.getModelVariant() == ModelConfig::ModelVariant::kGlm;
    auto const mropeRotaryCosSinSize = modelConfig.getMaxPositionEmbeddings() * modelConfig.getRotaryEmbeddingDim();

    {
        NVTX3_SCOPED_RANGE(seqSlotsLoop);
        auto* seqSlotIndices = bufferCast<SizeType32>(*seqSlots);

        SizeType32 batchIdx{0};
        for (auto const& requests : {contextRequests, genRequests})
        {
            for (auto const& llmReq : requests)
            {
                // Get position of the current sequence in the decoder
                for (auto const& seqSlot : llmReq->mSeqSlots)
                {
                    seqSlotIndices[batchIdx] = seqSlot;
                    ++batchIdx;
                }
            }
        }

        TLLM_CHECK(seqSlots->getSize() == static_cast<std::size_t>(batchIdx));
        manager.copy(*seqSlots, *seqSlotsDevice);
    }

    SizeType32 contextRequestsSize = 0;
    for (auto const& llmReq : contextRequests)
    {
        //auto numSeq = llmReq->getNumSequences();
        // For CFG requests, we process inputs twice (conditional + unconditional)
        //contextRequestsSize += numSeq * (llmReq->isCfg() ? 2 : 1);
        contextRequestsSize += llmReq->getNumSequences();
    }

    // context preparation loop
    if (contextRequestsSize > 0)
    {
        NVTX3_SCOPED_RANGE(contextPrepareLoop);
        numContextLogits.resize(contextRequestsSize);

        SizeType32 batchIdx{0};
        for (auto const& llmReq : contextRequests)
        {
            TLLM_CHECK_WITH_INFO(llmReq->isContextInitState() || llmReq->isDisaggGenerationTransmissionComplete(),
                "The request should be in context phase or disaggregated generation tranmissionComplete phase.");
            TLLM_CHECK_WITH_INFO(
                llmReq->getMaxNumGeneratedTokens() == 0, "Context request should not have generated tokens.");

            auto const& reqTokens = llmReq->getTokens(0);
            // for CFG requests, add the inputs to the buffer twice
            std::vector<bool> is_conditional_vec{true};
            if (llmReq->isCfg())
            {
                is_conditional_vec.push_back(false);
            }
            for (auto const& is_conditional : is_conditional_vec)
            {
                auto const& origTokens = llmReq->getTokens(0);
                std::vector<TokenIdType> dummyTokens;
                if (!is_conditional)
                {
                    // that is special token added in "convert_checkpoint",
                    // that is expanded to all zeros
                    dummyTokens.assign(origTokens.size(), modelConfig.getVocabSize());
                }

                auto const& reqTokens = is_conditional ? origTokens : dummyTokens;
                auto const& draftTokens = llmReq->getDraftTokens();
                auto const draftLength = llmReq->getNumDraftTokens();
                auto const& positionIds = llmReq->getPositionIds();

                auto const contextChunkSize = llmReq->getContextChunkSize();
                auto const beginCompute = llmReq->getContextCurrentPosition();
                auto const endCompute = beginCompute + contextChunkSize;
                auto const beginComputeFlat = beginCompute * llmReq->getNumVocabs();
                auto const contextChunkSizeFlat = contextChunkSize * llmReq->getNumVocabs();
                inputHost.insert(inputHost.end(), reqTokens.begin() + beginComputeFlat,
                    reqTokens.begin() + beginComputeFlat + contextChunkSizeFlat);

                logitsIdsHostPtr[totalNumLogits++] = contextChunkSize;
                numContextLogits.at(batchIdx) = modelConfig.computeContextLogits() ? contextChunkSize : 1;

                if (llmReq->isLastContextChunk())
                {
                    inputHost.insert(inputHost.end(), draftTokens->begin(), draftTokens->end());
                    std::fill_n(logitsIdsHostPtr + totalNumLogits, draftLength, 1);
                    totalNumLogits += draftLength;
                }
                auto const inputLength = contextChunkSize + (llmReq->isLastContextChunk() ? draftLength : 0);
                contextLengthsHostPtr[batchIdx] = inputLength;
                auto const sequenceLen = inputLength + llmReq->getContextCurrentPosition();
                sequenceLengthsHostPtr[batchIdx] = sequenceLen;

                if (static_cast<bool>(pastKeyValueLengthsPtr))
                {
                    pastKeyValueLengthsPtr[batchIdx] = beginCompute + inputLength;
                }

                if (positionIds.has_value())
                {
                    TLLM_CHECK_WITH_INFO(
                        !(isChatGlm || isGlm), "ChatGLM-6B and Glm only use the default initialization");
                    positionIdsHost.insert(positionIdsHost.end(), positionIds.value()->begin() + beginCompute,
                        positionIds.value()->begin() + endCompute);
                }
                else
                {
                    if (isChatGlm)
                    {
                        // Specialize for ChatGLM-6B with 2D-Position-Embedding
                        positionIdsHost.resize(totalInputSize + inputLength);
                        std::iota(std::begin(positionIdsHost) + totalInputSize, std::end(positionIdsHost), 0);
                        positionIdsHost.back() = positionIdsHost.back() - 1;

                        positionIdsHostRow2.resize(totalInputSize + inputLength);
                        positionIdsHostRow2.back() = 1;
                    }
                    else if (isGlm)
                    {
                        // Specialize for GLM-10B with 2D-Position-Embedding and special value of the mask id position
                        auto start = inputHost.begin() + totalInputSize;
                        auto end = start + inputLength;
                        auto it = std::find_if(
                            start, end, [](SizeType32 id) { return id == 50260 || id == 50263 || id == 50264; });
                        llmReq->mMaskPosition = (it != end) ? std::distance(start, it) : maxContextLength;

                        positionIdsHost.resize(totalInputSize + inputLength);
                        std::iota(std::begin(positionIdsHost) + totalInputSize, std::end(positionIdsHost), 0);
                        positionIdsHost.back() = llmReq->mMaskPosition;

                        positionIdsHostRow2.resize(totalInputSize + inputLength);
                        positionIdsHostRow2.back() = 1;
                    }
                    else
                    {
                        // Other models
                        positionIdsHost.resize(totalInputSize + inputLength);
                        std::iota(std::begin(positionIdsHost) + totalInputSize,
                            std::begin(positionIdsHost) + totalInputSize + inputLength, beginCompute);
                    }
                }
                if (modelConfig.useMrope())
                {
                    auto optMropeRotaryCosSin = llmReq->getMropeRotaryCosSin().value();
                    TLLM_CHECK_WITH_INFO(optMropeRotaryCosSin->getShape().d[0] == mropeRotaryCosSinSize,
                        "Provided MropeRotarySinCos is %ld and expected is %d.\n",
                        optMropeRotaryCosSin->getShape().d[0], int(mropeRotaryCosSinSize));

                    auto const mropeRotaryCosSinCtx = ITensor::slice(mropeRotaryCosSin, batchIdx, 1);
                    manager.copy(*optMropeRotaryCosSin, *mropeRotaryCosSinCtx);
                }

                if (modelConfig.useLanguageAdapter())
                {
                    auto const languageAdapterRouting = llmReq->getLanguageAdapterRouting(
                        modelConfig.getNumLanguages().value(), endCompute - beginCompute);
                    languageAdapterRoutingsHost.insert(languageAdapterRoutingsHost.end(),
                        std::begin(languageAdapterRouting), std::end(languageAdapterRouting));
                }
                totalInputSize += inputLength;
                ++batchIdx;
            }
        }

        if (rnnStateBuffers)
        {
            // TODO: dont implement CFG for rnn state buffers for now
            rnnStateBuffers->fillSlotMappings(contextRequests, rnnStateManagerPtr);
        }

        // set decoder context features and mask
        if (useContextEmbeddings)
        {
            SizeType32 tokenIdx = 0;
            for (auto const& llmReq : contextRequests)
            {
                auto const contextPosition = llmReq->getContextCurrentPosition();
                auto const contextChunkSize = llmReq->getContextChunkSize();
                if (llmReq->getDecoderContextFeatures())
                {
                    auto const& reqFeatures = llmReq->getDecoderContextFeatures();
                    auto const& reqFeaturesShape = reqFeatures->getShape();
                    TLLM_CHECK_WITH_INFO(reqFeaturesShape.nbDims >= 2,
                        "Decoder context features must have at least 2 dimensions, but got %d dimensions for request %lu",
                        reqFeaturesShape.nbDims, llmReq->mRequestId);
                    // If caller provides per-sequence context features laid out as
                    // [seq0_full][seq1_full]..., copy the active chunk for every CFG sequence.
                    // A single contiguous slice is only correct when contextPosition == 0 and the
                    // chunk spans the whole prompt; chunked context must stride by the per-sequence
                    // feature length.
                    auto const numSeqs = llmReq->getNumSequences();
                    auto const hasPerSeqFeatures = numSeqs > 1 && reqFeaturesShape.d[0] >= numSeqs * llmReq->mPromptLen;
                    auto const featuresToCopy = hasPerSeqFeatures ? numSeqs * contextChunkSize : contextChunkSize;
                    TLLM_CHECK_WITH_INFO(contextPosition + contextChunkSize <= reqFeaturesShape.d[0],
                        "Decoder context features [%d, %d], but request is at position %d and chunk size %d (numSeqs=%d)",
                        (int) reqFeaturesShape.d[0], (int) reqFeaturesShape.d[1], contextPosition,
                        contextChunkSize, numSeqs);
                    if (hasPerSeqFeatures)
                    {
                        auto const perSeqFeatureLen = reqFeaturesShape.d[0] / numSeqs;
                        TLLM_CHECK_WITH_INFO(contextPosition + contextChunkSize <= perSeqFeatureLen,
                            "Decoder context features per sequence length %d is too short for position %d and chunk "
                            "size %d",
                            (int) perSeqFeatureLen, contextPosition, contextChunkSize);
                        for (SizeType32 seqIdx = 0; seqIdx < numSeqs; ++seqIdx)
                        {
                            auto const srcOffset = seqIdx * perSeqFeatureLen + contextPosition;
                            auto const dstOffset = tokenIdx + seqIdx * contextChunkSize;
                            manager.copy(*ITensor::slice(reqFeatures, srcOffset, contextChunkSize),
                                *ITensor::slice(decoderContextFeatures, dstOffset, contextChunkSize));
                            manager.setMem(*ITensor::slice(decoderContextFeaturesMask, dstOffset, contextChunkSize), 1);
                        }
                    }
                    else
                    {
                        // Old behaviour for callers that only provide the conditional context chunk.
                        manager.copy(*ITensor::slice(reqFeatures, contextPosition, featuresToCopy),
                            *ITensor::slice(decoderContextFeatures, tokenIdx, featuresToCopy));
                        manager.setMem(*ITensor::slice(decoderContextFeaturesMask, tokenIdx, featuresToCopy), 1);
                    }
                }
                tokenIdx += llmReq->getNumSequences() * contextChunkSize;
            }
        }
    }

    // generation preparation loop - CHECK THIS LINE ONWARDS
    if (!genRequests.empty())
    {
        NVTX3_SCOPED_RANGE(genPrepareLoop);

        auto numSequences = contextRequestsSize;

        for (auto const& llmReq : genRequests)
        {
            for (int s = 0; s < llmReq->getNumSequences(); s++)
            {
                auto reqBeamWidth = llmReq->getBeamWidthByIter();
                auto const draftLength = llmReq->getNumDraftTokens();
                auto const& draftTokens = llmReq->getDraftTokens();
                auto const numLogits = draftLength + reqBeamWidth;
                TLLM_CHECK(draftLength == 0 || reqBeamWidth == 1);

                auto const promptLen = llmReq->mPromptLen;
                auto const sequenceLen
                    = promptLen + llmReq->getMaxNumGeneratedTokens() + static_cast<SizeType32>(trtOverlap);
                auto const& positionIds = llmReq->getPositionIds();
                for (int reqBeam = 0; reqBeam < reqBeamWidth; ++reqBeam)
                {
                    // for CFG, simply use tokens from the 0th beam during generation
                    int beam = llmReq->isCfg() ? 0 : reqBeam;
                    auto const numTokens = llmReq->getNumTokens(beam) + static_cast<SizeType32>(trtOverlap);
                    // TODO: can this be removed completely?
                    if (!trtOverlap)
                    {
                        if (llmReq->getNumVocabs() > 1)
                        {
                            auto const& beamTokens = llmReq->getTokens(beam);
                            TLLM_CHECK_WITH_INFO(beamTokens.size() % llmReq->getNumVocabs() == 0,
                                "Number of tokens needs to be a multiple of number of vocabs! %d %d", beamTokens.size(), llmReq->getNumVocabs());
                            inputHost.insert(
                                inputHost.end(), beamTokens.cend() - llmReq->getNumVocabs(), beamTokens.cend());
                        }
                        else
                        {
                            auto const lastToken = llmReq->getLastTokens(beam);
                            inputHost.push_back(lastToken);
                        }
                        if (draftLength > 0)
                        {
                            inputHost.insert(inputHost.end(), draftTokens->begin(), draftTokens->end());
                        }
                    }

                    // If model updates generation position ids do not append them here.
                    if (!modelConfig.getSpeculativeDecodingMode().updatesPositionIds())
                    {
                        if (positionIds.has_value())
                        {
                            TLLM_CHECK_WITH_INFO(
                                !(isChatGlm || isGlm), "ChatGLM-6B and Glm only use the default initialization");
                            auto last_context_position_id = positionIds.value()->back();
                            positionIdsHost.push_back(
                                static_cast<SizeType32>(last_context_position_id + sequenceLen - promptLen));
                        }
                        else
                        {
                            if (isChatGlm) // ChatGLM-6B
                            {
                                positionIdsHost.push_back(static_cast<SizeType32>(promptLen - 2));
                                positionIdsHostRow2.push_back(static_cast<SizeType32>(sequenceLen - promptLen + 1));
                            }
                            else if (isGlm)
                            {
                                positionIdsHost.push_back(llmReq->mMaskPosition);
                                positionIdsHostRow2.push_back(static_cast<SizeType32>(sequenceLen - promptLen + 1));
                            }
                            else // GPT / ChatGLM2-6B / ChatGLM3-6B / BART
                            {
                                // positionIds is just the size of tokens -1
                                positionIdsHost.push_back(numTokens - 1);
                            }
                        }
                    }

                    if (modelConfig.useMrope())
                    {
                        auto optMropePositionDeltas = llmReq->getMropePositionDeltas().value();
                        mropePositionDeltasHost.push_back(optMropePositionDeltas);
                    }

                    if (modelConfig.useLanguageAdapter())
                    {
                        // Generation requests only have one token per sequence
                        auto const languageAdapterRouting
                            = llmReq->getLanguageAdapterRouting(modelConfig.getNumLanguages().value(), 1);
                        languageAdapterRoutingsHost.insert(languageAdapterRoutingsHost.end(),
                            std::begin(languageAdapterRouting), std::end(languageAdapterRouting));
                    }
                }

                if (static_cast<bool>(pastKeyValueLengthsPtr))
                {
                    SizeType32 pastKeyValueLength = sequenceLen - 1;
                    std::fill_n(pastKeyValueLengthsPtr + numSequences, reqBeamWidth, pastKeyValueLength);
                }
                totalInputSize += numLogits;

                std::fill_n(logitsIdsHostPtr + totalNumLogits, numLogits, 1);

                totalNumLogits += numLogits;

                if (rnnStateBuffers)
                {
                    TLLM_CHECK_WITH_INFO(!llmReq->isCfg(), "CFG is not supported for rnn state buffers");
                    auto const seqSlot = llmReq->mSeqSlots[0];
                    auto& rnnStateManager = *rnnStateManagerPtr;
                    rnnStateManager.fillSlotMapping(
                        *rnnStateBuffers->slotMappingHost, numSequences, seqSlot, reqBeamWidth);
                }
                numSequences += reqBeamWidth;
            }
        }

        if (transformerBuffers && maxBeamWidth > 1)
        {
            transformerBuffers->copyCacheIndirection(genRequests, decoderState.getCacheIndirectionOutput(), stream);
        }

        numSequences = contextRequestsSize;
        for (auto const& llmReq : genRequests)
        {
            for (int s = 0; s < llmReq->getNumSequences(); s++)
            {
                auto const reqBeamWidth = llmReq->getBeamWidthByIter();
                auto const draftLength = llmReq->getNumDraftTokens();

                auto const contextQLength = llmReq->mPromptLen + draftLength;
                auto const sequenceLen
                    = contextQLength + llmReq->getMaxNumGeneratedTokens() + static_cast<SizeType32>(trtOverlap);

                std::fill_n(contextLengthsHostPtr + numSequences, reqBeamWidth, contextQLength);
                std::fill_n(sequenceLengthsHostPtr + numSequences, reqBeamWidth, sequenceLen);
                numSequences += reqBeamWidth;
            }
        }

        if (modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding())
        {
            // copy from lookahead decoding buffer
            mLookaheadBuffers->setFromInputs(numContextRequests, numGenRequests, *requestTypes, *seqSlots,
                decoderState.getLookaheadBuffers(), runtime, modelConfig, worldConfig);
        }

        if (useAttentionPrior)
        {
            // set to zero attention prior scores, so scores from different layers can be accumulated
            manager.setMem(*attentionPriorScores, 0);
            // copy focus indices from llm requests to a buffer.
            // Encoding: a non-negative value F means "apply mask centered at F, store scores at [F, F+lookahead)".
            // A negative value -(F+1) means "no mask on this step, but still store scores at [F, F+lookahead)" —
            // used for a request's very first generation step so the model's natural cross-attention
            // (un-distorted by the prior) is captured to seed the focus index from step 2 onwards.
            // This matches NeMo's behaviour where attn_prior=None on the first decoder step.
            // A value kSinkPriorFocusOffset + F means "apply a NeMo sink prior from F": suppress all history
            // before F and keep only [F, F+lookahead). This is used after a token has been attended
            // maxAttendCount times; NeMo sets attn_prior[:, :stuck_timestep+1] = epsilon in that case.
            constexpr int kSinkPriorFocusOffset = 1 << 29;
            auto const getInitialFocusAbs = [](int leftOffset, SizeType32 encoderOutputLen) {
                auto const maxLocalFocus = std::max(0, static_cast<int>(encoderOutputLen) - 1);
                if (leftOffset <= 0)
                {
                    constexpr int kDefaultFirstFocus = 4;
                    return std::min(kDefaultFirstFocus, maxLocalFocus);
                }
                return leftOffset + std::min(5, maxLocalFocus);
            };
            auto const toLocalFocus = [](int focusAbs, int leftOffset, SizeType32 encoderOutputLen) {
                auto const maxLocalFocus = std::max(0, static_cast<int>(encoderOutputLen) - 1);
                return std::min(std::max(0, focusAbs - leftOffset), maxLocalFocus);
            };
            // Context rows must not be prior-masked, especially in fused context+generation batches.
            // Use the same negative sentinel as first generation step: store scores, but do not mask.
            // The focus buffer is sequence-indexed, not request-indexed; CFG context requests therefore
            // need both conditional and unconditional entries.
            std::vector<int> focus_lst;
            focus_lst.reserve(getNumSequences());
            for (auto const& llmReq : contextRequests)
            {
                int const leftOffset = static_cast<int>(llmReq->getLeftOffset());
                int const initialFocusAbs = getInitialFocusAbs(leftOffset, llmReq->getEncoderOutputLen());
                int const initialFocusLocal = toLocalFocus(initialFocusAbs, leftOffset, llmReq->getEncoderOutputLen());
                int const encodedContext = -(initialFocusLocal + 1);
                for (SizeType32 i = 0; i < llmReq->getNumSequences(); i++)
                {
                    focus_lst.push_back(encodedContext);
                }
            }
            for (auto const& llmReq : genRequests)
            {
                bool const firstGenStep = !llmReq->hasAttentionPriorIdx();
                int const leftOffset = static_cast<int>(llmReq->getLeftOffset());
                int const initialFocusAbs = getInitialFocusAbs(leftOffset, llmReq->getEncoderOutputLen());
                int focusAbs = firstGenStep ? initialFocusAbs : static_cast<int>(llmReq->getAttentionPriorIdx(modelConfig));
                int const sinkAttendCount = std::max(1, llmReq->getMaxAttendCount());
                bool const sinkPrior = !firstGenStep
                    && static_cast<int>(llmReq->getAttentionPriorAttendCount()) >= sinkAttendCount
                    && llmReq->getEncoderOutputLen() > 0
                    && focusAbs < leftOffset + static_cast<int>(llmReq->getEncoderOutputLen()) - 1;
                if (sinkPrior)
                {
                    // NeMo applies this suppression to the current forward after the previous
                    // step increments the dwell count. The score window for this forward starts
                    // at the first non-suppressed token.
                    focusAbs += 1;
                }
                int const focusLocal = toLocalFocus(focusAbs, leftOffset, llmReq->getEncoderOutputLen());
                int const encodedCond
                    = firstGenStep ? -(focusLocal + 1) : (sinkPrior ? kSinkPriorFocusOffset + focusLocal : focusLocal);
                int const encodedUncond = -(focusLocal + 1);
                for (SizeType32 i = 0; i < llmReq->getNumSequences(); i++)
                {
                    // Under CFG, NeMo applies the attention prior only to the conditional half.
                    // The unconditional half receives an all-epsilon prior, which normalizes to no mask.
                    focus_lst.push_back(llmReq->isCfg() && i == 1 ? encodedUncond : encodedCond);
                }
            }
            manager.copy(focus_lst.data(), *attentionPriorFocus, runtime::MemoryType::kCPU);
        }
    }

    // check skipCrossAttnBlocks
    if (transformerBuffers && modelConfig.skipCrossAttnBlocks())
    {
        bool isSkipCrossAttn = true;
        for (auto const& requests : {contextRequests, genRequests})
        {
            for (auto const& llmReq : requests)
            {
                bool tmpValue = false;
                if (llmReq->getSkipCrossAttnBlocks() != nullptr)
                {
                    manager.copy(*llmReq->getSkipCrossAttnBlocks(), &tmpValue);
                }
                isSkipCrossAttn &= tmpValue;
            }
        }
        transformerBuffers->copySkipCrossAttnBlocks(isSkipCrossAttn, runtime);
    }

    if (isChatGlm || isGlm)
    {
        positionIdsHost.reserve(totalInputSize * 2);
        positionIdsHost.insert(positionIdsHost.end(), positionIdsHostRow2.begin(), positionIdsHostRow2.end());
    }

    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->fill(contextRequests, genRequests, manager);
    }
    if (modelConfig.usePromptTuning())
    {
        promptTuningBuffers->fill(contextRequests, genRequests, manager, modelConfig.usePackedInput());
    }
    if (modelConfig.useLoraPlugin())
    {
        loraBuffers->fill(contextRequests, genRequests, peftTable, manager, modelConfig, worldConfig);
    }
    if (modelConfig.useMrope())
    {
        if (!mropePositionDeltasHost.empty())
        {
            auto mropePositionDeltasGen = ITensor::slice(mropePositionDeltas, 0, numGenSequences);
            manager.copy(mropePositionDeltasHost.data(), *mropePositionDeltasGen);
        }
    }

    {
        NVTX3_SCOPED_RANGE(bufferCopies);
        if (trtOverlap)
        {
            auto contextInputsIds = ITensor::slice(inputsIds, 0, numContextTokens);
            manager.copy(inputHost.data(), *contextInputsIds);

            if (!genRequests.empty())
            {
                auto generationInputsIds = ITensor::slice(inputsIds, numContextTokens);
                auto seqSlotsDeviceSlice = ITensor::slice(seqSlotsDevice, numContextRequests);
                runtime::kernels::invokeGatherBatch(
                    *generationInputsIds, *newOutputTokens, *seqSlotsDeviceSlice, maxBeamWidth, stream);
            }
        }
        else
        {
            manager.copy(inputHost.data(), *inputsIds);
        }
        // In generation phase, device ptr of context lengths need to be tiled.
        manager.copy(*contextLengthsHost, *contextLengthsDevice);
        manager.copy(*sequenceLengthsHost, *sequenceLengthsDevice);
        auto const logitsIdsHostRange = BufferRange<SizeType32>(*logitsIdsHost);
        auto lastTokenIdsHostRange = BufferRange<SizeType32>(*lastTokenIdsHost);
        common::stl_utils::inclusiveScan(
            logitsIdsHostRange.begin(), logitsIdsHostRange.end(), lastTokenIdsHostRange.begin());
        manager.copy(*lastTokenIdsHost, *lastTokenIdsDevice);
        if (transformerBuffers)
        {
            TensorPtr decoderPositionIds = modelConfig.getSpeculativeDecodingMode().isLookaheadDecoding()
                ? mLookaheadBuffers->positionIdsDevice
                : nullptr;
            transformerBuffers->copyPositionIds(runtime, positionIdsHost, isChatGlm || isGlm, decoderPositionIds);
        }
        if (rnnStateBuffers)
        {
            rnnStateBuffers->copySlotMappingH2D(runtime);
        }
        if (modelConfig.useLanguageAdapter())
        {
            manager.copy(languageAdapterRoutingsHost.data(), *languageAdapterRoutings);
        }
    }

    if (transformerBuffers && static_cast<bool>(kvCacheManagerPtr))
    {
        transformerBuffers->copyKvBlockOffsets(
            contextRequests, genRequests, kvCacheManagerPtr, crossKvCacheManagerPtr, manager);
    }

    if (modelConfig.useCrossAttention())
    {
        transformerBuffers->copyCrossAttentionMasks(contextRequests, genRequests, contextLengthsDevice,
            encoderBuffers->inputLengths, maxContextLength, encoderBuffers->getMaxInputLengthInBatch(), runtime);
    }

    maxKvCacheLengthRounded = 0;
    if (static_cast<bool>(pastKeyValueLengthsPtr))
    {
        auto const maxKvCacheLength
            = *std::max_element(pastKeyValueLengthsPtr, pastKeyValueLengthsPtr + getNumSequences());
        // Round up kv cache length
        maxKvCacheLengthRounded = common::ceilDiv(maxKvCacheLength, kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE)
            * kKV_CACHE_LEN_CUDA_GRAPH_ROUND_SIZE;
    }

    if (modelConfig.getSpeculativeDecodingMode().needsDecoderPrologue())
    {
        if (modelConfig.getSpeculativeDecodingMode().isExplicitDraftTokens())
        {
            prepareExplicitDraftTokenBuffers(
                decoderState.getExplicitDraftTokensBuffers(), runtime, modelConfig, worldConfig);
        }
        if (modelConfig.getSpeculativeDecodingMode().isEagle())
        {
            prepareEagleBuffers(
                contextRequests, genRequests, decoderState.getEagleBuffers(), runtime, modelConfig, worldConfig);
        }
    }

    sync_check_cuda_error(stream.get());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareExplicitDraftTokenBuffers(
    runtime::ExplicitDraftTokensBuffers::Inputs const& explicitDraftTokensBuffers, TllmRuntime const& runtime,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(mExplicitDraftTokensBuffers);

    mExplicitDraftTokensBuffers->setFromInputs(numContextRequests, numGenRequests, *requestTypes, *seqSlots,
        explicitDraftTokensBuffers, *transformerBuffers->positionIds, modelConfig, worldConfig,
        runtime.getBufferManager(), runtime.getStream());

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::prepareEagleBuffers(RequestVector const& contextRequests, RequestVector const& genRequests,
    runtime::EagleBuffers::Inputs const& eagleBuffers, TllmRuntime const& runtime, ModelConfig const& modelConfig,
    WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);

    TLLM_CHECK(mEagleBuffers);

    mEagleBuffers->setFromInputs(contextRequests, genRequests, *requestTypes, *seqSlots, eagleBuffers,
        runtime.getBufferManager(), modelConfig, worldConfig);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

void RuntimeBuffers::processAttentionPriorScores(RequestVector const& contextRequests, RequestVector const& genRequests,
    TllmRuntime const& runtime, ModelConfig const& modelConfig)
{
    /**
     * is called after inference is done. processes the "scores" buffer and sets up
     * the index with most attention focus for each request.
     */
    if (!useAttentionPrior)
    {
        TLLM_LOG_WARNING("processing attention prior scores, when attention prior is disabled");
        return;
    }

    // copy scores to host
    auto const& manager = runtime.getBufferManager();
    auto const& stream = runtime.getStream();
    auto scoresHost
        = manager.cpu(ITensor::makeShape({getNumSequences() * attentionPriorLookahead}), nvinfer1::DataType::kFLOAT);
    manager.copy(*attentionPriorScores, *scoresHost);
    stream.synchronize();

    auto* scoresHostPtr = bufferCast<float>(*scoresHost);

    ReqIdsSet contextSeededRequests;
    size_t scoresOffset = 0;
    auto rowHasSignal = [&](size_t rowOffset) {
        for (int i = 0; i < attentionPriorLookahead; i++)
        {
            if (scoresHostPtr[rowOffset + i] != 0.0F)
            {
                return true;
            }
        }
        return false;
    };
    auto selectScoreRow = [&](size_t baseOffset, SizeType32 numSequences) {
        size_t selectedOffset = baseOffset;
        int selectedSequence = 0;
        if (!rowHasSignal(baseOffset))
        {
            for (SizeType32 seqIdx = 1; seqIdx < numSequences; seqIdx++)
            {
                auto const rowOffset = baseOffset + static_cast<size_t>(seqIdx) * attentionPriorLookahead;
                if (rowHasSignal(rowOffset))
                {
                    selectedOffset = rowOffset;
                    selectedSequence = static_cast<int>(seqIdx);
                    break;
                }
            }
        }
        return std::make_pair(selectedOffset, selectedSequence);
    };

    // Magpie's Riva path bakes the generated-audio BOS into decoder_context_features.
    // That makes the context/prompt row equivalent to NeMo's first decoder step
    // (attn_prior=None), so seed the prior from the context row.
    for (auto const& llmReq : contextRequests)
    {
        if (!llmReq->hasAttentionPriorIdx())
        {
            auto const leftOffset = static_cast<size_t>(llmReq->getLeftOffset());
            auto const encoderOutputLen = llmReq->getEncoderOutputLen();
            auto const maxLocalFocus = encoderOutputLen > 0 ? static_cast<size_t>(encoderOutputLen - 1) : 0;
            constexpr int kDefaultFirstFocus = 4;
            auto const initialFocus = leftOffset > 0
                ? leftOffset + std::min<size_t>(5, maxLocalFocus)
                : std::min(static_cast<size_t>(kDefaultFirstFocus), maxLocalFocus);
            size_t const prevPriorIdx = initialFocus;

            int candidateLimit = attentionPriorLookahead;
            bool terminalFallback = false;
            if (encoderOutputLen > 5)
            {
                auto const terminalWindowStart = leftOffset + static_cast<size_t>(encoderOutputLen - 3);
                if (prevPriorIdx >= terminalWindowStart)
                {
                    terminalFallback = true;
                    candidateLimit = 0;
                }
                else
                {
                    candidateLimit = std::min<int>(
                        attentionPriorLookahead, static_cast<int>(terminalWindowStart - prevPriorIdx));
                }
            }

            auto const [selectedScoresOffset, selectedSequence]
                = selectScoreRow(scoresOffset, llmReq->getNumSequences());
            auto* selectedScores = scoresHostPtr + selectedScoresOffset;

            float maxScore = selectedScores[0];
            int idxShift = 0;
            if (terminalFallback || candidateLimit <= 0)
            {
                auto const finalIdx = encoderOutputLen > 0
                    ? leftOffset + static_cast<size_t>(encoderOutputLen - 1)
                    : prevPriorIdx;
                idxShift = finalIdx > prevPriorIdx ? static_cast<int>(finalIdx - prevPriorIdx) : 0;
                auto const scoreIdx = std::min(idxShift, attentionPriorLookahead - 1);
                maxScore = selectedScores[scoreIdx];
            }
            else
            {
                for (int i = 1; i < candidateLimit; i++)
                {
                    if (selectedScores[i] > maxScore)
                    {
                        maxScore = selectedScores[i];
                        idxShift = i;
                    }
                }
            }

            constexpr float kContextMinConfidence = 0.01F;
            bool hasContextSignal = false;
            for (int i = 0; i < attentionPriorLookahead; i++)
            {
                hasContextSignal = hasContextSignal || selectedScores[i] != 0.0F;
            }
            if (selectedSequence != 0 && maxScore < kContextMinConfidence)
            {
                hasContextSignal = false;
            }
            if (hasContextSignal)
            {
                auto const newPriorIdx = prevPriorIdx + idxShift;
                llmReq->setAttentionPriorIdx(newPriorIdx, modelConfig);
                if (char const* tracePath = std::getenv("TRT_EOS_POLICY_TRACE");
                    tracePath != nullptr && tracePath[0] != '\0')
                {
                    std::ofstream out(tracePath, std::ios::app);
                    out << "attn_seed"
                        << "\treq=" << llmReq->mRequestId
                        << "\tseq=" << selectedSequence
                        << "\tprev=" << prevPriorIdx
                        << "\tshift=" << idxShift
                        << "\tnew=" << newPriorIdx
                        << "\tleft_offset=" << leftOffset
                        << "\tenc_len=" << encoderOutputLen
                        << "\tmax_score=" << maxScore
                        << "\tcandidate_limit=" << candidateLimit
                        << "\tterminal=" << (terminalFallback ? 1 : 0)
                        << "\tnear_end=" << (llmReq->isAttentionPriorNearEnd() ? 1 : 0)
                        << "\tfinished=" << (llmReq->isAttentionPriorFinished() ? 1 : 0)
                        << "\tend_count=" << llmReq->getAttentionPriorEndAttendCount()
                        << "\n";
                }
                contextSeededRequests.insert(llmReq->mRequestId);
            }
        }
        scoresOffset += attentionPriorLookahead * llmReq->getNumSequences();
    }

    // for each generation request, analyze scores and set the attention prior idx
    for (auto const& llmReq : genRequests)
    {
        if (contextSeededRequests.count(llmReq->mRequestId) != 0)
        {
            // The generation row in this same forward pass was computed before the context row
            // seeded the request. Leave the context seed in place for the next iteration.
            scoresOffset += attentionPriorLookahead * llmReq->getNumSequences();
            continue;
        }

        // First-gen-step seeding: if no focus was set yet, the kernel ran without masking
        // (sentinel = -(initial+1)) and wrote scores at [initial, initial+lookahead).
        // Use the argmax of those scores to seed focus, exactly the way subsequent steps work.
        bool const firstPriorStep = !llmReq->hasAttentionPriorIdx();
        auto const leftOffset = static_cast<size_t>(llmReq->getLeftOffset());
        auto const encoderOutputLen = llmReq->getEncoderOutputLen();
        auto const maxLocalFocus = encoderOutputLen > 0 ? static_cast<size_t>(encoderOutputLen - 1) : 0;
        constexpr int kDefaultFirstFocus = 4;
        auto const initialFocus = leftOffset > 0
            ? leftOffset + std::min<size_t>(5, maxLocalFocus)
            : std::min(static_cast<size_t>(kDefaultFirstFocus), maxLocalFocus);
        size_t prevPriorIdx = firstPriorStep ? initialFocus : llmReq->getAttentionPriorIdx(modelConfig);
        int const dwellAtPrev = static_cast<int>(llmReq->getAttentionPriorAttendCount());
        auto const selectedScoresOffset = selectScoreRow(scoresOffset, llmReq->getNumSequences()).first;
        auto* selectedScores = scoresHostPtr + selectedScoresOffset;

        // Match NeMo's get_most_attended_text_timestep ordering: first check whether the
        // previously attended timestep has become a sink, then select the next attended
        // timestep from the lookahead window. Do not advance again after incrementing
        // the counter for the newly selected timestep.
        int const sinkAttendCount = std::max(1, llmReq->getMaxAttendCount());
        if (!firstPriorStep && dwellAtPrev >= sinkAttendCount && encoderOutputLen > 0
            && prevPriorIdx < leftOffset + static_cast<size_t>(encoderOutputLen - 1))
        {
            prevPriorIdx += 1;
        }

        // Match NeMo's get_most_attended_text_timestep terminal-window rule:
        // normal argmax searches [prevFocus, min(prevFocus + lookahead, text_len - 3)).
        // Once that slice is empty, NeMo jumps to text_len - 1 and lets the
        // finished-text counter hold the end before EOS. Without this exclusion,
        // low-confidence tail probabilities can choose text_len - 2 / text_len - 1
        // too early and skip the last phones of long utterances.
        int candidateLimit = attentionPriorLookahead;
        bool terminalFallback = false;
        if (encoderOutputLen > 5)
        {
            auto const terminalWindowStart = leftOffset + static_cast<size_t>(encoderOutputLen - 3);
            if (prevPriorIdx >= terminalWindowStart)
            {
                terminalFallback = true;
                candidateLimit = 0;
            }
            else
            {
                candidateLimit = std::min<int>(
                    attentionPriorLookahead, static_cast<int>(terminalWindowStart - prevPriorIdx));
            }
        }

        float maxScore = selectedScores[0];
        int idxShift = 0;
        if (terminalFallback || candidateLimit <= 0)
        {
            auto const finalIdx = encoderOutputLen > 0
                ? leftOffset + static_cast<size_t>(encoderOutputLen - 1)
                : prevPriorIdx;
            idxShift = finalIdx > prevPriorIdx ? static_cast<int>(finalIdx - prevPriorIdx) : 0;
            auto const scoreIdx = std::min(idxShift, attentionPriorLookahead - 1);
            maxScore = selectedScores[scoreIdx];
        }
        else
        {
            for (int i = 1; i < candidateLimit; i++)
            {
                if (selectedScores[i] > maxScore)
                {
                    maxScore = selectedScores[i];
                    idxShift = i;
                }
            }
        }

        // Flat-region forced advance. When the lookahead distribution is too flat to trust
        // (peak < minConf), the argmax is noise: a small/zero idxShift under-advances and can
        // stall/repeat. Force only those under-advances up to a minimal movement.
        // NOTE: do NOT scale this by frame_stacking_factor. NeMo's reference advances the focus by
        // argmax over the lookahead and only nudges +1 on attention sinks (get_most_attended); it
        // never force-advances by the stacking factor. Default flatAdvance = 1 to match NeMo; the
        // focus still keeps pace via the argmax-based advance + the per-position sink-advance.
        // Tunables: TRT_ATTN_PRIOR_MIN_CONF (default 0 = disabled, matching NeMo),
        //           TRT_ATTN_PRIOR_MIN_ADVANCE (default = 1; override to experiment).
        static float const minConf = []() {
            char const* e = std::getenv("TRT_ATTN_PRIOR_MIN_CONF");
            return e != nullptr ? static_cast<float>(std::atof(e)) : 0.0f;
        }();
        int const minAdvance = []() {
            char const* e = std::getenv("TRT_ATTN_PRIOR_MIN_ADVANCE");
            return e != nullptr ? std::atoi(e) : -1;
        }();
        int const flatAdvance = minAdvance > 0 ? minAdvance : 1;
        // Only force-move the focus when the prior is actually being applied. On the first/sentinel
        // gen step the kernel runs WITHOUT the mask (apply_prior_mask=false) to capture the model's
        // natural attention for seeding — forcing there would corrupt that seed. So gate on
        // !firstPriorStep (prior is set/applied this step). (And processAttentionPriorScores already
        // early-returns when useAttentionPrior is false, so this never runs in no-prior mode.)
        bool forcedAdvance = false;
        if (!terminalFallback && !firstPriorStep && minConf > 0.0f && maxScore < minConf && idxShift < flatAdvance)
        {
            idxShift = flatAdvance;
            forcedAdvance = true;
        }

        // Tail guard: near the end of long text, a low-confidence argmax several positions ahead
        // can skip the last phonemes/words and sound like longform drift. Keep this opt-in while
        // validating: it caps only large forward jumps in the final tail window.
        static int const tailWindow = []() {
            char const* e = std::getenv("TRT_ATTN_PRIOR_TAIL_MAX_ADVANCE_WINDOW");
            return e != nullptr ? std::atoi(e) : 0;
        }();
        static float const tailConf = []() {
            char const* e = std::getenv("TRT_ATTN_PRIOR_TAIL_MAX_ADVANCE_CONF");
            return e != nullptr ? static_cast<float>(std::atof(e)) : 0.0f;
        }();
        static int const tailMaxAdvance = []() {
            char const* e = std::getenv("TRT_ATTN_PRIOR_TAIL_MAX_ADVANCE");
            return e != nullptr ? std::max(1, std::atoi(e)) : 1;
        }();
        auto const finalIdx = encoderOutputLen > 0
            ? leftOffset + static_cast<size_t>(encoderOutputLen - 1)
            : prevPriorIdx;
        auto const remainingToEnd
            = encoderOutputLen > 0 && prevPriorIdx <= finalIdx
            ? static_cast<int>(finalIdx - prevPriorIdx)
            : 0;
        if (!terminalFallback && !firstPriorStep && !forcedAdvance && tailWindow > 0 && tailConf > 0.0f
            && remainingToEnd <= tailWindow && maxScore < tailConf && idxShift > tailMaxAdvance)
        {
            idxShift = tailMaxAdvance;
            forcedAdvance = true;
        }

        // Anti-dwell: a CONFIDENT peak pinned on the current focus (idxShift==0 with maxScore>=minConf)
        // is a stutter the gate above cannot catch — it only fires on LOW confidence. Attention-score
        // analysis of Hindi/number dwells shows the argmax re-selecting the current position (lookahead
        // index 0) for several steps at high maxScore (~0.4-0.8), re-uttering the same audio. If this
        // position has already been held >= maxDwell steps, force-advance regardless of confidence.
        // Env TRT_ATTN_PRIOR_MAX_DWELL (default 0 = disabled -> max_attend_count stays the only cap).
        // Tune carefully: too small clips legitimately sustained phonemes (a held vowel spans several
        // frame-stacked steps at the same focus). This is independent of, and fires earlier than, the
        // max_attend_count sink-bump at the beginning of this function.
        static int const maxDwell = []() {
            char const* e = std::getenv("TRT_ATTN_PRIOR_MAX_DWELL");
            return e != nullptr ? std::atoi(e) : 0;
        }();
        if (!terminalFallback && !firstPriorStep && !forcedAdvance && maxDwell > 0 && idxShift == 0 && dwellAtPrev >= maxDwell)
        {
            idxShift = flatAdvance;
            forcedAdvance = true;
        }

        // LlmRequest::setAttentionPriorIdx() only records the selected attended position.
        // The NeMo-style sink-bump happens before argmax above on the next decoder step.
        auto const newPriorIdx = prevPriorIdx + idxShift;
        llmReq->setAttentionPriorIdx(newPriorIdx, modelConfig);
        if (char const* tracePath = std::getenv("TRT_EOS_POLICY_TRACE");
            tracePath != nullptr && tracePath[0] != '\0')
        {
            std::ofstream out(tracePath, std::ios::app);
            out << "attn_update"
                << "\treq=" << llmReq->mRequestId
                << "\tfirst=" << (firstPriorStep ? 1 : 0)
                << "\tprev=" << prevPriorIdx
                << "\tdwell_prev=" << dwellAtPrev
                << "\tshift=" << idxShift
                << "\tnew=" << newPriorIdx
                << "\tleft_offset=" << leftOffset
                << "\tenc_len=" << encoderOutputLen
                << "\tmax_score=" << maxScore
                << "\tcandidate_limit=" << candidateLimit
                << "\tterminal=" << (terminalFallback ? 1 : 0)
                << "\tforced=" << (forcedAdvance ? 1 : 0)
                << "\tnear_end=" << (llmReq->isAttentionPriorNearEnd() ? 1 : 0)
                << "\tfinished=" << (llmReq->isAttentionPriorFinished() ? 1 : 0)
                << "\tend_count=" << llmReq->getAttentionPriorEndAttendCount()
                << "\tmax_end=" << llmReq->getMaxEndAttendCount()
                << "\n";
        }

        // TODO: remove hardcode of lookahead size
        scoresOffset += attentionPriorLookahead * llmReq->getNumSequences();
    }
}

std::tuple<SizeType32, RuntimeBuffers::TensorMap const&, RuntimeBuffers::TensorMap&> RuntimeBuffers::prepareStep(
    RequestVector const& contextRequests, RequestVector const& genRequests, SizeType32 maxBeamWidth,
    SizeType32 maxAttentionWindow, runtime::decoder::DecoderState const& decoderState,
    kv_cache_manager::BaseKVCacheManager* kvCacheManager, kv_cache_manager::BaseKVCacheManager* crossKvCacheManager,
    rnn_state_manager::RnnStateManager* rnnStateManager, PeftTable const& peftTable, TllmRuntime const& runtime,
    ModelConfig const& modelConfig, WorldConfig const& worldConfig, bool gatherGenerationLogits, bool trtOverlap,
    OptionalRef<runtime::ITensor const> newOutputTokens)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersPrepareStep);

    setBufferSizes(contextRequests, genRequests);
    reshape(runtime, modelConfig, worldConfig, gatherGenerationLogits);

    setFromInputs(contextRequests, genRequests, maxBeamWidth, maxAttentionWindow, decoderState, kvCacheManager,
        crossKvCacheManager, rnnStateManager, peftTable, runtime, modelConfig, worldConfig, trtOverlap,
        newOutputTokens);

    fillIOMaps(modelConfig, worldConfig);

    auto const numTokens = getNumTokens();
    auto const optProfileId = runtime.getOptProfileId(numTokens, ModelConfig::getOptProfilesSplitPoints());
    setContextIndex(optProfileId);
    TLLM_LOG_DEBUG("numTokens: %d, optProfileId: %d", numTokens, optProfileId);

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return {optProfileId, inputMap, outputMap};
}

void RuntimeBuffers::fillIOMaps(ModelConfig const& modelConfig, WorldConfig const& worldConfig)
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    NVTX3_SCOPED_RANGE(runtimeBuffersFillIOMaps);

    inputMap.clear();
    outputMap.clear();

    if (transformerBuffers)
    {
        transformerBuffers->getBuffers(inputMap, outputMap, modelConfig);
    }
    if (rnnStateBuffers)
    {
        rnnStateBuffers->getBuffers(inputMap);
    }

    if (worldConfig.isLastPipelineParallelRank())
    {
        // feed a view to TensorRT runtime so reshaping does not change logits buffer
        outputMap.insert_or_assign(kLogitsTensorName, ITensor::view(logits));
    }
    else
    {
        outputMap.insert_or_assign(kHiddenStatesOutputTensorName, hiddenStates);
    }

    if (worldConfig.isFirstPipelineParallelRank())
    {
        inputMap.insert_or_assign(kInputIdsTensorName, inputsIds);
    }
    else
    {
        inputMap.insert_or_assign(kHiddenStatesInputTensorName, hiddenStates);
    }

    inputMap.insert_or_assign(kLastTokenIdsTensorName, lastTokenIdsDevice);

    inputMap.insert_or_assign(kHostRequestTypesTensorName, requestTypes);
    // In the generation phase, we still pass context lengths.
    inputMap.insert_or_assign(kContextLengthsTensorName, contextLengthsDevice);
    inputMap.insert_or_assign(kHostContextLengthsTensorName, contextLengthsHost);
    inputMap.insert_or_assign(kSequenceLengthsTensorName, sequenceLengthsDevice);

    if (useContextEmbeddings)
    {
        inputMap.insert_or_assign(kDecoderContextFeaturesTensorName, decoderContextFeatures);
        inputMap.insert_or_assign(kDecoderContextFeaturesMaskTensorName, decoderContextFeaturesMask);
    }
    if (useAttentionPrior)
    {
        inputMap.insert_or_assign(kAttentionPriorFocusTensorName, attentionPriorFocus);
    }
    if (modelConfig.useCrossAttention())
    {
        encoderBuffers->insertInputTensors(inputMap);
    }
    if (modelConfig.usePromptTuning())
    {
        auto const& promptTuningParams = promptTuningBuffers->mPromptTuningParams;
        inputMap.insert_or_assign(kPromptEmbeddingTableTensorName, promptTuningParams.embeddingTable);
        inputMap.insert_or_assign(kTasksTensorName, promptTuningParams.tasks);
        inputMap.insert_or_assign(kPromptVocabSizeTensorName, promptTuningParams.vocabSize);
    }
    if (modelConfig.useMrope())
    {

        inputMap.insert_or_assign(kMRopeRotaryCosSinTensorName, mropeRotaryCosSin);
        inputMap.insert_or_assign(kMRopePositionDeltasTensorName, mropePositionDeltas);
    }
    if (modelConfig.useLoraPlugin())
    {
        loraBuffers->insertInputTensors(inputMap, loraBuffers->mLoraWeightsPointersHost,
            loraBuffers->mLoraAdapterSizesHost, modelConfig, worldConfig);
    }
    if (modelConfig.useLanguageAdapter())
    {
        inputMap.insert_or_assign("language_adapter_routings", languageAdapterRoutings);
    }

    if (mMedusaBuffers)
    {
        mMedusaBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }
    if (mLookaheadBuffers)
    {
        mLookaheadBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }
    if (mExplicitDraftTokensBuffers)
    {
        mExplicitDraftTokensBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }
    if (mEagleBuffers)
    {
        mEagleBuffers->insertInputTensors(inputMap, outputMap, worldConfig);
    }
    if (useAttentionPrior)
    {
        outputMap.insert_or_assign(kAttentionPriorScoresTensorName, attentionPriorScores);
    }

    for (auto const& outputTensor : mAdditionalOutputTensors)
    {
        outputMap.insert_or_assign(outputTensor.first, outputTensor.second);
    }

    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

} // namespace tensorrt_llm::batch_manager
