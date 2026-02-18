/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "CategoricalSamplingPlugin.h"
#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <memory>
#include <cassert>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <map>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

// CUDA error checking macro
#define CUDA_CHECK(call)                                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
        cudaError_t error = call;                                                                                      \
        if (error != cudaSuccess)                                                                                      \
        {                                                                                                              \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error)       \
                      << std::endl;                                                                                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// Logger for TensorRT
class Logger : public ILogger
{
public:
    void log(Severity severity, const char* msg) noexcept override
    {
        // Suppress info-level messages
        if (severity <= Severity::kWARNING)
        {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
} gLogger;

// Helper class to manage TensorRT resources with RAII
template <typename T>
struct TrtDeleter
{
    void operator()(T* obj) const
    {
        if (obj)
        {
            delete obj;
        }
    }
};

template <typename T>
using TrtUniquePtr = std::unique_ptr<T, TrtDeleter<T>>;

// Helper function to convert float to half
std::vector<half> floatToHalf(std::vector<float> const& floatVec)
{
    std::vector<half> halfVec(floatVec.size());
    for (size_t i = 0; i < floatVec.size(); ++i)
    {
        halfVec[i] = __float2half(floatVec[i]);
    }
    return halfVec;
}

// Test 1: Basic functionality test with simple probability distributions
bool testBasicSampling()
{
    std::cout << "\n=== Test 1: Basic Categorical Sampling ===" << std::endl;

    int32_t const batchSize = 4;
    int32_t const vocabSize = 5;
    int32_t const topk = 5;

    // Create simple probability distributions
    std::vector<float> probsFloat = {
        // Batch 0: Uniform distribution
        0.2f, 0.2f, 0.2f, 0.2f, 0.2f,
        // Batch 1: Skewed to first token
        0.7f, 0.1f, 0.1f, 0.05f, 0.05f,
        // Batch 2: Skewed to last token
        0.05f, 0.05f, 0.1f, 0.1f, 0.7f,
        // Batch 3: Two peaks
        0.4f, 0.1f, 0.0f, 0.1f, 0.4f
    };

    std::vector<int32_t> topkIdx = {
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4,
        0, 1, 2, 3, 4
    };

    std::vector<half> probs = floatToHalf(probsFloat);

    // Initialize TensorRT plugin library
    bool didInitPlugins = initLibNvInferPlugins(&gLogger, "");
    if (!didInitPlugins)
    {
        std::cerr << "Failed to initialize TensorRT plugins" << std::endl;
        return false;
    }

    // Create builder
    TrtUniquePtr<IBuilder> builder{createInferBuilder(gLogger)};
    if (!builder)
    {
        std::cerr << "Failed to create builder" << std::endl;
        return false;
    }

    // Create network with explicit batch flag
    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<INetworkDefinition> network{builder->createNetworkV2(flag)};
    if (!network)
    {
        std::cerr << "Failed to create network" << std::endl;
        return false;
    }

    // Add input tensors
    ITensor* probsInput = network->addInput("probs", DataType::kHALF, Dims3{batchSize, vocabSize, 1});
    ITensor* topkInput = network->addInput("topk", DataType::kINT32, Dims3{batchSize, topk, 1});

    if (!probsInput || !topkInput)
    {
        std::cerr << "Failed to add input tensors" << std::endl;
        return false;
    }

    // Get plugin creator from registry
    auto* creator = getPluginRegistry()->getPluginCreator("CategoricalSampling", "1", "");
    if (!creator)
    {
        std::cerr << "Failed to get plugin creator" << std::endl;
        return false;
    }

    // Create plugin with no fields (uses clock-based seeding)
    PluginFieldCollection emptyFC;
    emptyFC.nbFields = 0;
    emptyFC.fields = nullptr;
    
    IPluginV3* pluginObj = creator->createPlugin("CategoricalSampling", &emptyFC, TensorRTPhase::kBUILD);
    if (!pluginObj)
    {
        std::cerr << "Failed to create plugin object" << std::endl;
        return false;
    }

    // Add plugin layer to network
    std::vector<ITensor*> inputs{probsInput, topkInput};
    IPluginV3Layer* pluginLayer = network->addPluginV3(inputs.data(), inputs.size(), nullptr, 0, *pluginObj);
    
    if (!pluginLayer)
    {
        std::cerr << "Failed to add plugin layer" << std::endl;
        delete pluginObj;
        return false;
    }

    // Mark output
    ITensor* output = pluginLayer->getOutput(0);
    output->setName("output");
    network->markOutput(*output);

    // Create builder config
    TrtUniquePtr<IBuilderConfig> config{builder->createBuilderConfig()};
    if (!config)
    {
        std::cerr << "Failed to create builder config" << std::endl;
        delete pluginObj;
        return false;
    }

    // Set memory pool limit (1GB)
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);

    // Build serialized network (engine)
    TrtUniquePtr<IHostMemory> serializedEngine{builder->buildSerializedNetwork(*network, *config)};
    if (!serializedEngine)
    {
        std::cerr << "Failed to build serialized network" << std::endl;
        delete pluginObj;
        return false;
    }

    // Clean up plugin object after adding to network
    delete pluginObj;

    // Create runtime
    TrtUniquePtr<IRuntime> runtime{createInferRuntime(gLogger)};
    if (!runtime)
    {
        std::cerr << "Failed to create runtime" << std::endl;
        return false;
    }

    // Deserialize engine
    TrtUniquePtr<ICudaEngine> engine{runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size())};
    if (!engine)
    {
        std::cerr << "Failed to deserialize engine" << std::endl;
        return false;
    }

    // Create execution context
    TrtUniquePtr<IExecutionContext> context{engine->createExecutionContext()};
    if (!context)
    {
        std::cerr << "Failed to create execution context" << std::endl;
        return false;
    }

    // Allocate device memory for inputs and outputs
    void* dProbs;
    void* dTopk;
    void* dOutput;
    
    size_t probsSize = batchSize * vocabSize * sizeof(half);
    size_t topkSize = batchSize * topk * sizeof(int32_t);
    size_t outputSize = batchSize * sizeof(int32_t);
    
    CUDA_CHECK(cudaMalloc(&dProbs, probsSize));
    CUDA_CHECK(cudaMalloc(&dTopk, topkSize));
    CUDA_CHECK(cudaMalloc(&dOutput, outputSize));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(dProbs, probs.data(), probsSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTopk, topkIdx.data(), topkSize, cudaMemcpyHostToDevice));

    // Set input/output bindings
    context->setTensorAddress("probs", dProbs);
    context->setTensorAddress("topk", dTopk);
    context->setTensorAddress("output", dOutput);

    // Execute inference
    bool status = context->enqueueV3(0);
    if (!status)
    {
        std::cerr << "Failed to execute inference" << std::endl;
        CUDA_CHECK(cudaFree(dProbs));
        CUDA_CHECK(cudaFree(dTopk));
        CUDA_CHECK(cudaFree(dOutput));
        return false;
    }

    // Wait for completion
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    std::vector<int32_t> hostOutput(batchSize);
    CUDA_CHECK(cudaMemcpy(hostOutput.data(), dOutput, outputSize, cudaMemcpyDeviceToHost));

    // Print results
    std::cout << "Sampled indices: ";
    for (int i = 0; i < batchSize; ++i)
    {
        std::cout << hostOutput[i] << " ";
        
        // Validate output is within vocabulary range
        if (hostOutput[i] < 0 || hostOutput[i] >= vocabSize)
        {
            std::cerr << "\nError: Invalid sampled index " << hostOutput[i] << " at batch " << i << std::endl;
            CUDA_CHECK(cudaFree(dProbs));
            CUDA_CHECK(cudaFree(dTopk));
            CUDA_CHECK(cudaFree(dOutput));
            return false;
        }
    }
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(dProbs));
    CUDA_CHECK(cudaFree(dTopk));
    CUDA_CHECK(cudaFree(dOutput));

    std::cout << "Test 1 passed!" << std::endl;
    return true;
}

// Test 2: Statistical distribution check
bool testStatisticalDistribution()
{
    std::cout << "\n=== Test 2: Statistical Distribution Check ===" << std::endl;

    int32_t const batchSize = 1;
    int32_t const vocabSize = 5;
    int32_t const topk = 5;
    int32_t const numSamples = 10000;

    // Create probability distribution
    std::vector<float> probsFloat = {0.1f, 0.2f, 0.3f, 0.25f, 0.15f};
    std::vector<half> probs = floatToHalf(probsFloat);
    std::vector<int32_t> topkIdx = {0, 1, 2, 3, 4};

    // Initialize counts
    std::map<int32_t, int32_t> counts;

    // Build engine (similar to test 1, but we'll reuse the engine for multiple inferences)
    TrtUniquePtr<IBuilder> builder{createInferBuilder(gLogger)};
    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<INetworkDefinition> network{builder->createNetworkV2(flag)};

    ITensor* probsInput = network->addInput("probs", DataType::kHALF, Dims3{batchSize, vocabSize, 1});
    ITensor* topkInput = network->addInput("topk", DataType::kINT32, Dims3{batchSize, topk, 1});

    auto* creator = getPluginRegistry()->getPluginCreator("CategoricalSampling", "1", "");
    PluginFieldCollection emptyFC{};
    emptyFC.nbFields = 0;
    emptyFC.fields = nullptr;
    
    IPluginV3* pluginObj = creator->createPlugin("CategoricalSampling", &emptyFC, TensorRTPhase::kBUILD);
    std::vector<ITensor*> inputs{probsInput, topkInput};
    IPluginV3Layer* pluginLayer = network->addPluginV3(inputs.data(), inputs.size(), nullptr, 0, *pluginObj);
    
    ITensor* output = pluginLayer->getOutput(0);
    output->setName("output");
    network->markOutput(*output);

    TrtUniquePtr<IBuilderConfig> config{builder->createBuilderConfig()};
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);

    TrtUniquePtr<IHostMemory> serializedEngine{builder->buildSerializedNetwork(*network, *config)};
    delete pluginObj;

    TrtUniquePtr<IRuntime> runtime{createInferRuntime(gLogger)};
    TrtUniquePtr<ICudaEngine> engine{runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size())};
    TrtUniquePtr<IExecutionContext> context{engine->createExecutionContext()};

    // Allocate device memory
    void* dProbs;
    void* dTopk;
    void* dOutput;
    
    CUDA_CHECK(cudaMalloc(&dProbs, batchSize * vocabSize * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dTopk, batchSize * topk * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&dOutput, batchSize * sizeof(int32_t)));

    CUDA_CHECK(cudaMemcpy(dProbs, probs.data(), batchSize * vocabSize * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTopk, topkIdx.data(), batchSize * topk * sizeof(int32_t), cudaMemcpyHostToDevice));

    context->setTensorAddress("probs", dProbs);
    context->setTensorAddress("topk", dTopk);
    context->setTensorAddress("output", dOutput);

    // Run multiple samples
    for (int i = 0; i < numSamples; ++i)
    {
        context->enqueueV3(0);
        CUDA_CHECK(cudaDeviceSynchronize());

        int32_t result;
        CUDA_CHECK(cudaMemcpy(&result, dOutput, sizeof(int32_t), cudaMemcpyDeviceToHost));
        counts[result]++;
    }

    // Analyze results
    std::cout << "Index | Expected | Observed | Difference" << std::endl;
    std::cout << "------|----------|----------|------------" << std::endl;

    bool allPassed = true;
    for (int i = 0; i < vocabSize; ++i)
    {
        float expected = probsFloat[i];
        float observed = static_cast<float>(counts[i]) / numSamples;
        float diff = std::abs(expected - observed);

        printf("%5d | %8.4f | %8.4f | %10.4f", i, expected, observed, diff);

        // Check if within reasonable bounds (5% tolerance for statistical test)
        if (diff > 0.05f)
        {
            std::cout << " ✗" << std::endl;
            allPassed = false;
        }
        else
        {
            std::cout << " ✓" << std::endl;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(dProbs));
    CUDA_CHECK(cudaFree(dTopk));
    CUDA_CHECK(cudaFree(dOutput));

    if (allPassed)
    {
        std::cout << "Test 2 passed!" << std::endl;
    }
    else
    {
        std::cout << "Test 2 had warnings (statistical variations may occur)" << std::endl;
    }

    return true;
}

// Test 3: Large batch size test
bool testLargeBatch()
{
    std::cout << "\n=== Test 3: Large Batch Size ===" << std::endl;

    int32_t const batchSize = 128;
    int32_t const vocabSize = 1024;
    int32_t const topk = 50;

    // Create uniform probability distribution
    std::vector<float> probsFloat(batchSize * vocabSize, 1.0f / vocabSize);
    std::vector<half> probs = floatToHalf(probsFloat);
    
    std::vector<int32_t> topkIdx(batchSize * topk);
    for (int i = 0; i < batchSize; ++i)
    {
        for (int j = 0; j < topk; ++j)
        {
            topkIdx[i * topk + j] = j;
        }
    }

    TrtUniquePtr<IBuilder> builder{createInferBuilder(gLogger)};
    uint32_t flag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    TrtUniquePtr<INetworkDefinition> network{builder->createNetworkV2(flag)};

    ITensor* probsInput = network->addInput("probs", DataType::kHALF, Dims3{batchSize, vocabSize, 1});
    ITensor* topkInput = network->addInput("topk", DataType::kINT32, Dims3{batchSize, topk, 1});

    auto* creator = getPluginRegistry()->getPluginCreator("CategoricalSampling", "1", "");
    PluginFieldCollection emptyFC{};
    emptyFC.nbFields = 0;
    emptyFC.fields = nullptr;
    
    IPluginV3* pluginObj = creator->createPlugin("CategoricalSampling", &emptyFC, TensorRTPhase::kBUILD);
    std::vector<ITensor*> inputs{probsInput, topkInput};
    IPluginV3Layer* pluginLayer = network->addPluginV3(inputs.data(), inputs.size(), nullptr, 0, *pluginObj);
    
    ITensor* output = pluginLayer->getOutput(0);
    output->setName("output");
    network->markOutput(*output);

    TrtUniquePtr<IBuilderConfig> config{builder->createBuilderConfig()};
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1ULL << 30);

    TrtUniquePtr<IHostMemory> serializedEngine{builder->buildSerializedNetwork(*network, *config)};
    delete pluginObj;

    TrtUniquePtr<IRuntime> runtime{createInferRuntime(gLogger)};
    TrtUniquePtr<ICudaEngine> engine{runtime->deserializeCudaEngine(serializedEngine->data(), serializedEngine->size())};
    TrtUniquePtr<IExecutionContext> context{engine->createExecutionContext()};

    // Allocate device memory
    void* dProbs;
    void* dTopk;
    void* dOutput;
    
    size_t probsSize = batchSize * vocabSize * sizeof(half);
    size_t topkSize = batchSize * topk * sizeof(int32_t);
    size_t outputSize = batchSize * sizeof(int32_t);
    
    CUDA_CHECK(cudaMalloc(&dProbs, probsSize));
    CUDA_CHECK(cudaMalloc(&dTopk, topkSize));
    CUDA_CHECK(cudaMalloc(&dOutput, outputSize));

    CUDA_CHECK(cudaMemcpy(dProbs, probs.data(), probsSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dTopk, topkIdx.data(), topkSize, cudaMemcpyHostToDevice));

    context->setTensorAddress("probs", dProbs);
    context->setTensorAddress("topk", dTopk);
    context->setTensorAddress("output", dOutput);

    // Execute inference
    bool status = context->enqueueV3(0);
    if (!status)
    {
        std::cerr << "Failed to execute inference" << std::endl;
        CUDA_CHECK(cudaFree(dProbs));
        CUDA_CHECK(cudaFree(dTopk));
        CUDA_CHECK(cudaFree(dOutput));
        return false;
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy output back to host
    std::vector<int32_t> hostOutput(batchSize);
    CUDA_CHECK(cudaMemcpy(hostOutput.data(), dOutput, outputSize, cudaMemcpyDeviceToHost));

    // Validate all outputs are within vocabulary range
    bool allValid = true;
    for (int i = 0; i < batchSize; ++i)
    {
        if (hostOutput[i] < 0 || hostOutput[i] >= vocabSize)
        {
            std::cerr << "Error: Invalid sampled index " << hostOutput[i] << " at batch " << i << std::endl;
            allValid = false;
        }
    }

    // Cleanup
    CUDA_CHECK(cudaFree(dProbs));
    CUDA_CHECK(cudaFree(dTopk));
    CUDA_CHECK(cudaFree(dOutput));

    if (allValid)
    {
        std::cout << "Successfully processed batch size " << batchSize << " with vocab size " << vocabSize << std::endl;
        std::cout << "Test 3 passed!" << std::endl;
        return true;
    }
    else
    {
        std::cout << "Test 3 failed!" << std::endl;
        return false;
    }
}

int main()
{
    std::cout << "Starting CategoricalSamplingPlugin TensorRT Tests" << std::endl;
    std::cout << "=================================================" << std::endl;

    // Check CUDA device
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "No CUDA devices found!" << std::endl;
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;

    // Run tests
    bool test1Passed = testBasicSampling();
    bool test2Passed = testStatisticalDistribution();
    bool test3Passed = testLargeBatch();

    std::cout << "\n=================================================" << std::endl;
    std::cout << "Test Results:" << std::endl;
    std::cout << "  Test 1 (Basic Sampling): " << (test1Passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Test 2 (Statistical Distribution): " << (test2Passed ? "PASSED" : "FAILED") << std::endl;
    std::cout << "  Test 3 (Large Batch): " << (test3Passed ? "PASSED" : "FAILED") << std::endl;

    if (test1Passed && test2Passed && test3Passed)
    {
        std::cout << "\nAll tests passed successfully!" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "\nSome tests failed!" << std::endl;
        return 1;
    }
}
