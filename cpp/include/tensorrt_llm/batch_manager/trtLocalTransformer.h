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

#pragma once

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/decodingOutput.h"
#include "tensorrt_llm/runtime/gptDecoderBatched.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/rawEngine.h"
#include "tensorrt_llm/runtime/tllmRuntime.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>
#include <memory>

namespace tensorrt_llm::runtime
{
class GptDecoderBatched;

namespace decoder
{
class DecoderState;
}
}

namespace tensorrt_llm::batch_manager
{
class DecoderInputBuffers;
class DecoderOutputBuffers;
class SlotDecoderBuffers;
class MakeDecodingBatchInputOutput;

class TrtLocalTransformer
{
public:
    using TensorPtr = runtime::ITensor::SharedPtr;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using TensorMap = runtime::ITensor::TensorMap;

    static constexpr auto kInHiddenStatesTensorName = "hidden_states";
    static constexpr auto kRandUniformTensorName = "rand_uniform";
    static constexpr auto kEosPolicyTensorName = "eos_policy";
    static constexpr auto kOutLogitsTensorName = "logits";

    TrtLocalTransformer(
        runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig,
        runtime::RawEngine const& rawEngine,
        SizeType32 numVocabs,  // Number of decoder buffers to create (typically 1-8, NOT the vocabulary size)
        std::shared_ptr<nvinfer1::ILogger> logger,
        SizeType32 maxNumSequences,
        SizeType32 maxSequenceLen,
        SizeType32 numMicroBatches,
        SizeType32 maxBatchSize);
    
    ~TrtLocalTransformer();

    /// \brief Run the local transformer using hiddenStates and current request sets.
    void run(TensorPtr const& hiddenStates,
        RequestVector const& contextRequests,
        std::vector<SizeType32> const& numContextFramesVec,
        RequestVector const& generationRequests,
        SizeType32 microBatchId,
        SizeType32 fusedBufferId
    );

    [[nodiscard]] std::shared_ptr<runtime::GptDecoderBatched>& getDecoder();
    [[nodiscard]] DecoderInputBuffers& getDecoderInputBuffers(SizeType32 microBatchId);
    [[nodiscard]] std::shared_ptr<runtime::decoder::DecoderState>& getDecoderBuffers(SizeType32 vocabId);
    [[nodiscard]] std::shared_ptr<SlotDecoderBuffers>& getSlotDecoderBuffers(SizeType32 seqSlot);

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const;
    [[nodiscard]] runtime::BufferManager::CudaStreamPtr getRuntimeStreamPtr() const;
    [[nodiscard]] runtime::CudaStream const& getRuntimeStream() const;
    [[nodiscard]] SizeType32 getVocabSize();
    [[nodiscard]] SizeType32 getNumTokens();

    [[nodiscard]] TensorPtr getOutLogitsHost();
    void clearDecoderBuffers();
    
    /// \brief Update decoder state (sequenceLengths, finishReasons) after token generation
    void updateDecoderStateAfterGeneration(
        RequestVector const& contextRequests,
        RequestVector const& generationRequests,
        TensorPtr const& generatedTokens
    );
    
    /// \brief Get host buffers for decoder output (populated by updateDecoderStateAfterGeneration)
    [[nodiscard]] TensorPtr getSequenceLengthsHost() const;
    [[nodiscard]] TensorPtr getFinishReasonsHost() const;
    [[nodiscard]] TensorPtr getFinishedSumHost() const;

private:
    runtime::ModelConfig mModelConfig;
    runtime::WorldConfig mWorldConfig;
    int mDevice{-1};
    std::shared_ptr<runtime::TllmRuntime> mRuntime;
    int hiddenSize;
    int numTokens;
    int vocabSize;

    TensorPtr inHiddenStates;  // [batch x dim]
    TensorPtr inHiddenStatesHost;  // [batch x dim]
    TensorPtr inRandUniform;  // [numTokens x batch]
    TensorPtr inRandUniformHost;  // [numTokens x batch]
    TensorPtr inEosPolicy;  // [batch]
    TensorPtr inEosPolicyHost;  // [batch]
    TensorPtr inTokens;  // [8 x batch']
    TensorPtr inTokensSliceHost;  // [batch']
    TensorPtr outLogitsHost;  // [batch' x VocabSize]
    TensorPtr outLogits;  // [batch' x VocabSize]
    TensorMap inputMap;
    TensorMap outputMap;

    // Static engine descriptors cached at construction to avoid per-step engine queries.
    nvinfer1::DataType mHiddenStatesType{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType mRandUniformType{nvinfer1::DataType::kFLOAT};
    nvinfer1::DataType mEosPolicyType{nvinfer1::DataType::kFLOAT};
    bool mHasEosPolicyInput{false};

    // Pre-allocated GPU/CPU buffers at max-batch size to avoid per-step allocations in run().
    // Sized for maxBatchSize*2 rows (CFG doubles the batch).
    TensorPtr mInHiddenStatesBuf;
    TensorPtr mInEosPolicyBuf;
    TensorPtr mInEosPolicyHostBuf;
    TensorPtr mOutLogitsBuf;
    TensorPtr mOutLogitsHostBuf;
    TensorPtr mReorderedStatesBuf;
    SizeType32 mMaxBatchSize{0};

    // decoder (sampler) and buffers
    SizeType32 mMaxNumSequences;
    std::shared_ptr<runtime::GptDecoderBatched> mDecoder;
    std::vector<std::unique_ptr<runtime::decoder_batch::Input>> mDecodingInputs;
    std::unique_ptr<runtime::DecodingOutput> mDecodingOutput;
    std::vector<DecoderInputBuffers> mDecoderInputBuffers;
    std::vector<std::shared_ptr<runtime::decoder::DecoderState>> mDecoderBuffers;
    std::vector<std::shared_ptr<SlotDecoderBuffers>> mSlotDecoderBuffers;
    std::unique_ptr<tensorrt_llm::batch_manager::MakeDecodingBatchInputOutput const> mMakeDecodingBatchInputOutput;
    
    // Host buffers for decoder outputs (updated by updateDecoderStateAfterGeneration)
    TensorPtr mSequenceLengthsHost;
    TensorPtr mFinishReasonsHost;
    TensorPtr mFinishedSumHost;

    void HandleLogits(
        RequestVector const& contextRequests,
        RequestVector const& generationRequests
    );

};

} // namespace tensorrt_llm::batch_manager
