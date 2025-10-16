#pragma once

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include <cublas_v2.h>
#include <cuda_fp16.h>

namespace tensorrt_llm::kernels
{

//! Apply classifier-free guidance (CFG) on GPU in-place using cuBLAS.
//! It overwrites `logitsView` with: logits = cfgScale * logits + (1 - cfgScale) * uncondLogits
//! Only the slice [vocabOffset, vocabOffset + vocabSize) is modified.
inline void invokeCfg(tensorrt_llm::runtime::CudaStream const& stream, runtime::ITensor::SharedPtr logitsView,
    runtime::ITensor::SharedPtr uncondLogitsView, float cfgScale, runtime::SizeType32 vocabOffset,
    runtime::SizeType32 vocabSize)
{
    using TensorPtr = runtime::ITensor::SharedPtr;

    // Restrict to current vocabulary segment.
    TensorPtr logitsVocabView = runtime::ITensor::slice(logitsView, {0, vocabOffset}, vocabSize);
    TensorPtr uncondLogitsVocabView = runtime::ITensor::slice(uncondLogitsView, {0, vocabOffset}, vocabSize);

    void* condPtr = logitsVocabView->data();
    void const* uncondPtr = uncondLogitsVocabView->data();

    cudaDataType_t dataType{};
    switch (logitsVocabView->getDataType())
    {
    case nvinfer1::DataType::kFLOAT: dataType = CUDA_R_32F; break;
    case nvinfer1::DataType::kHALF: dataType = CUDA_R_16F; break;
    default: TLLM_THROW("Unsupported data type for CFG");
    }

    auto handlePtr = getCublasHandle();
    auto& handle = *handlePtr;
    tensorrt_llm::common::check_cuda_error(cublasSetStream(handle, stream.get()));

    int n = static_cast<int>(vocabSize);
    int inc = 1;

    // Use float for the scaling factors and always accumulate in FP32 to
    // satisfy cuBLAS requirements (FP16 vectors must use FP32 compute/alpha).
    float alphaF = cfgScale;                                                            // Scaling factor in FP32
    float axpyF = 1.0f - cfgScale;                                                      // (1 - cfgScale) in FP32

    tensorrt_llm::common::check_cuda_error(cublasScalEx(handle, n, &alphaF, CUDA_R_32F, // alpha
        condPtr, dataType,                                                              // x and its type
        inc, CUDA_R_32F));                                                              // increments + compute type

    tensorrt_llm::common::check_cuda_error(cublasAxpyEx(handle, n, &axpyF, CUDA_R_32F,  // alpha
        uncondPtr, dataType, inc,                                                       // x
        condPtr, dataType, inc,                                                         // y
        CUDA_R_32F));                                                                   // compute type
}

} // namespace tensorrt_llm::kernels
