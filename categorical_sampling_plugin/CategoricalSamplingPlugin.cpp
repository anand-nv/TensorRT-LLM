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
#include "categorical_sampling.cuh"
#include <cassert>
#include <cstring>
#include <cuda_fp16.h>
#include <iostream>
#include <random>

using namespace nvinfer1;
using namespace nvinfer1::plugin;

namespace
{
constexpr char const* CATEGORICAL_SAMPLING_PLUGIN_VERSION{"1"};
constexpr char const* CATEGORICAL_SAMPLING_PLUGIN_NAME{"CategoricalSampling"};
} // namespace

// Initialize static members
PluginFieldCollection CategoricalSamplingPluginCreator::mFC{};
std::vector<PluginField> CategoricalSamplingPluginCreator::mPluginAttributes;

//
// CategoricalSamplingPlugin
//

CategoricalSamplingPlugin::CategoricalSamplingPlugin()
    : mNamespace("")
{
}

CategoricalSamplingPlugin::CategoricalSamplingPlugin(CategoricalSamplingPlugin const& other)
    : mNamespace(other.mNamespace)
{
}

// IPluginV3 methods
IPluginCapability* CategoricalSamplingPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    try
    {
        switch (type)
        {
        case PluginCapabilityType::kBUILD: return static_cast<IPluginV3OneBuild*>(this);
        case PluginCapabilityType::kRUNTIME: return static_cast<IPluginV3OneRuntime*>(this);
        case PluginCapabilityType::kCORE: return static_cast<IPluginV3OneCore*>(this);
        default: return nullptr;
        }
    }
    catch (std::exception const& e)
    {
        std::cerr << "CategoricalSamplingPlugin::getCapabilityInterface: " << e.what() << std::endl;
    }
    return nullptr;
}

IPluginV3* CategoricalSamplingPlugin::clone() noexcept
{
    try
    {
        auto* plugin = new CategoricalSamplingPlugin(*this);
        return plugin;
    }
    catch (std::exception const& e)
    {
        std::cerr << "CategoricalSamplingPlugin::clone: " << e.what() << std::endl;
    }
    return nullptr;
}

// IPluginV3OneCore methods
char const* CategoricalSamplingPlugin::getPluginName() const noexcept
{
    return CATEGORICAL_SAMPLING_PLUGIN_NAME;
}

char const* CategoricalSamplingPlugin::getPluginVersion() const noexcept
{
    return CATEGORICAL_SAMPLING_PLUGIN_VERSION;
}

char const* CategoricalSamplingPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

// IPluginV3OneBuild methods
int32_t CategoricalSamplingPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t CategoricalSamplingPlugin::configurePlugin(
    DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

bool CategoricalSamplingPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    try
    {
        assert(nbInputs == 1);
        assert(nbOutputs == 1);

        if (pos == 0) // Probs input - only FP16
        {
            return (inOut[pos].desc.type == DataType::kHALF) && (inOut[pos].desc.format == TensorFormat::kLINEAR);
        }
        else if (pos == 1) // Output
        {
            return (inOut[pos].desc.type == DataType::kINT32) && (inOut[pos].desc.format == TensorFormat::kLINEAR);
        }
    }
    catch (std::exception const& e)
    {
        std::cerr << "CategoricalSamplingPlugin::supportsFormatCombination: " << e.what() << std::endl;
    }
    return false;
}

int32_t CategoricalSamplingPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    try
    {
        assert(nbOutputs == 1);
        assert(nbInputs == 1);
        assert(inputTypes[0] == DataType::kHALF);

        outputTypes[0] = DataType::kINT32;
        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr << "CategoricalSamplingPlugin::getOutputDataTypes: " << e.what() << std::endl;
        return -1;
    }
}

int32_t CategoricalSamplingPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs,
    DimsExprs const* shapeInputs, int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs,
    IExprBuilder& exprBuilder) noexcept
{
    try
    {
        assert(nbOutputs == 1);
        assert(nbShapeInputs == 0);
        assert(nbInputs == 1);

        auto const probsDims = inputs[0];
        assert(probsDims.nbDims == 2); // [batch_size, vocab_size]

        // Output shape is [batch_size]
        outputs[0].nbDims = 1;
        outputs[0].d[0] = probsDims.d[0];
        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr << "CategoricalSamplingPlugin::getOutputShapes: " << e.what() << std::endl;
        return -1;
    }
}

size_t CategoricalSamplingPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept
{
    // No workspace needed - the kernel handles memory internally
    return 0;
}

int32_t CategoricalSamplingPlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
{
    return 0;
}

int32_t CategoricalSamplingPlugin::getNbTactics() noexcept
{
    return 0;
}

char const* CategoricalSamplingPlugin::getTimingCacheID() noexcept
{
    return nullptr;
}

int32_t CategoricalSamplingPlugin::getFormatCombinationLimit() noexcept
{
    return 1;
}

char const* CategoricalSamplingPlugin::getMetadataString() noexcept
{
    return nullptr;
}

// IPluginV3OneRuntime methods
int32_t CategoricalSamplingPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept
{
    try
    {
        int32_t const batchSize = inputDesc[0].dims.d[0];
        int32_t const vocabSize = inputDesc[0].dims.d[1];
        half const* probs = static_cast<half const*>(inputs[0]);
        int32_t* output = static_cast<int32_t*>(outputs[0]);

#ifndef NDEBUG
        // Debug code - only compiled in debug builds
        std::cout << "batchSize: " << batchSize << ", vocabSize: " << vocabSize << std::endl;
        std::cout << "=== Input Tensor Debug ===" << std::endl;
        std::cout << "Input tensor has " << inputDesc[0].dims.nbDims << " dimensions: [";
        for (int i = 0; i < inputDesc[0].dims.nbDims; ++i)
        {
            std::cout << inputDesc[0].dims.d[i];
            if (i < inputDesc[0].dims.nbDims - 1)
                std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Interpreting as: batchSize=" << batchSize << ", vocabSize=" << vocabSize << std::endl;

        // Debug: Print probability statistics for first batch element
        std::vector<half> firstBatchProbs(vocabSize);
        cudaMemcpy(firstBatchProbs.data(), probs, vocabSize * sizeof(half), cudaMemcpyDeviceToHost);

        std::cout << "First batch - first 20 values: ";
        for (int i = 0; i < std::min(20, vocabSize); ++i)
        {
            std::cout << __half2float(firstBatchProbs[i]) << " ";
        }
        std::cout << std::endl;

        // Find max value and its index
        float maxVal = -1e10f;
        int maxIdx = -1;
        float sum = 0.0f;
        for (int i = 0; i < vocabSize; ++i)
        {
            float val = __half2float(firstBatchProbs[i]);
            sum += val;
            if (val > maxVal)
            {
                maxVal = val;
                maxIdx = i;
            }
        }
        std::cout << "First batch stats: max_value=" << maxVal << " at index=" << maxIdx << ", sum=" << sum
                  << ", mean=" << (sum / vocabSize) << std::endl;
#endif

        // Generate fresh entropy per enqueue so sampling is non-deterministic
        // across calls and across process restarts.
        static thread_local std::mt19937_64 rng{std::random_device{}()};
        unsigned long long const host_seed = rng();
        categoricalSampling(probs, output, batchSize, vocabSize, stream, host_seed);

        // Check for kernel errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            std::cerr << "CUDA kernel launch error: " << cudaGetErrorString(err) << std::endl;
            return -1;
        }

#ifndef NDEBUG
        // Debug: Copy output to host and print (only in debug builds)
        cudaDeviceSynchronize();
        std::vector<int32_t> hostOutput(batchSize);
        cudaMemcpy(hostOutput.data(), output, batchSize * sizeof(int32_t), cudaMemcpyDeviceToHost);
        std::cout << "Output batchSize: " << batchSize << std::endl;
        std::cout << "Sampled output: ";
        for (int i = 0; i < batchSize; i++)
        {
            std::cout << "i: " << i << ", output: " << hostOutput[i] << "\t";
        }
        std::cout << std::endl;
        std::cout << "========================" << std::endl;
#endif

        return 0;
    }
    catch (std::exception const& e)
    {
        std::cerr << "CategoricalSamplingPlugin::enqueue: " << e.what() << std::endl;
        return -1;
    }
}

int32_t CategoricalSamplingPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept
{
    return 0;
}

IPluginV3* CategoricalSamplingPlugin::attachToContext(IPluginResourceContext* context) noexcept
{
    return clone();
}

PluginFieldCollection const* CategoricalSamplingPlugin::getFieldsToSerialize() noexcept
{
    // No fields to serialize - plugin has no state
    return nullptr;
}

int32_t CategoricalSamplingPlugin::setTactic(int32_t tactic) noexcept
{
    return 0;
}

//
// CategoricalSamplingPluginCreator
//

CategoricalSamplingPluginCreator::CategoricalSamplingPluginCreator()
    : mNamespace("")
{
    // No plugin attributes needed - uses clock-based seeding
    mPluginAttributes.clear();
    mFC.nbFields = 0;
    mFC.fields = nullptr;
}

char const* CategoricalSamplingPluginCreator::getPluginName() const noexcept
{
    return CATEGORICAL_SAMPLING_PLUGIN_NAME;
}

char const* CategoricalSamplingPluginCreator::getPluginVersion() const noexcept
{
    return CATEGORICAL_SAMPLING_PLUGIN_VERSION;
}

PluginFieldCollection const* CategoricalSamplingPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

IPluginV3* CategoricalSamplingPluginCreator::createPlugin(
    char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept
{
    try
    {
        // No parameters needed - uses clock-based seeding
        auto* plugin = new CategoricalSamplingPlugin();
        return plugin;
    }
    catch (std::exception const& e)
    {
        std::cerr << "CategoricalSamplingPluginCreator::createPlugin: " << e.what() << std::endl;
    }
    return nullptr;
}

char const* CategoricalSamplingPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void CategoricalSamplingPluginCreator::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = pluginNamespace;
}

// Plugin registration - this makes the plugin available to TensorRT
REGISTER_TENSORRT_PLUGIN(CategoricalSamplingPluginCreator);
