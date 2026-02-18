# CategoricalSamplingPlugin TensorRT Test

This directory contains comprehensive tests for the `CategoricalSamplingPlugin` implementation using TensorRT's official API and best practices.

## Overview

The test suite validates the `CategoricalSamplingPlugin` by:
1. Creating TensorRT networks with the plugin
2. Building and serializing TensorRT engines
3. Executing inference with various input configurations
4. Validating outputs and statistical properties

## Test Implementation Details

The test implementation (`test_plugin.cpp`) follows TensorRT's official documentation and includes:

### Test Architecture

Based on [TensorRT's Extending Custom Layers documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html), the test:

1. **Network Creation**: Uses `INetworkDefinition` with explicit batch mode
2. **Plugin Integration**: Adds plugin using `addPluginV3()` API
3. **Engine Building**: Creates serialized engines with `buildSerializedNetwork()`
4. **Runtime Execution**: Uses `IExecutionContext` for inference
5. **Resource Management**: Employs RAII patterns with smart pointers

### Test Cases

#### Test 1: Basic Categorical Sampling
- **Purpose**: Validates basic functionality with simple probability distributions
- **Configuration**:
  - Batch size: 4
  - Vocabulary size: 5
  - Top-k: 5
- **Test Distributions**:
  - Uniform distribution (0.2, 0.2, 0.2, 0.2, 0.2)
  - Skewed to first token (0.7, 0.1, 0.1, 0.05, 0.05)
  - Skewed to last token (0.05, 0.05, 0.1, 0.1, 0.7)
  - Two peaks (0.4, 0.1, 0.0, 0.1, 0.4)
- **Validation**: Ensures all sampled indices are within valid vocabulary range

#### Test 2: Statistical Distribution Check
- **Purpose**: Validates that sampling follows the input probability distribution
- **Configuration**:
  - Batch size: 1
  - Vocabulary size: 5
  - Top-k: 5
  - Number of samples: 10,000
- **Test Distribution**: (0.1, 0.2, 0.3, 0.25, 0.15)
- **Validation**: 
  - Runs 10,000 sampling operations
  - Compares observed frequencies with expected probabilities
  - Tolerance: 5% deviation (accounts for statistical variation)
- **Note**: Uses clock-based seeding for non-reproducible randomness

#### Test 3: Large Batch Size Test
- **Purpose**: Validates scalability with larger inputs
- **Configuration**:
  - Batch size: 128
  - Vocabulary size: 1,024
  - Top-k: 50
- **Test Distribution**: Uniform distribution
- **Validation**: 
  - Ensures all 128 outputs are within vocabulary range
  - Tests GPU memory handling
  - Validates parallel execution

## Building and Running

### Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (11.0 or later)
- TensorRT (8.0 or later recommended, supports IPluginV3 interface)
- CMake (3.18 or later)
- C++ compiler with C++17 support

### Build Instructions

#### Option 1: Using the Build Script (Recommended)

```bash
cd categorical_sampling_plugin
./build_test.sh
```

The script will:
1. Automatically detect TensorRT installation
2. Configure and build the plugin library
3. Build the test executable
4. Optionally run the test

#### Option 2: Using Makefile (Simple)

```bash
cd categorical_sampling_plugin

# Build with default paths
make -f Makefile.test

# Or specify custom TensorRT path
make -f Makefile.test TENSORRT_ROOT=/path/to/tensorrt

# Build and run test
make -f Makefile.test test

# Clean
make -f Makefile.test clean
```

#### Option 3: Using CMake (Manual)

```bash
cd categorical_sampling_plugin
mkdir -p build_test
cd build_test

# Copy CMakeLists
cp ../CMakeLists_test.txt CMakeLists.txt

# Configure (adjust TENSORRT_ROOT if needed)
cmake -DTENSORRT_ROOT=/usr/local/tensorrt \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# Build
make -j$(nproc)

# Run
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/lib
./bin/test_plugin
```

### TensorRT Installation Paths

The build system searches for TensorRT in:
- `/usr/local/tensorrt`
- `/usr/local/TensorRT`
- `/opt/tensorrt`
- `$HOME/TensorRT`

If your TensorRT is installed elsewhere, set `TENSORRT_ROOT`:

```bash
export TENSORRT_ROOT=/path/to/your/tensorrt
./build_test.sh
```

## Expected Output

### Successful Test Run

```
Starting CategoricalSamplingPlugin TensorRT Tests
=================================================
Using GPU: <Your GPU Name>
Compute capability: X.X

=== Test 1: Basic Categorical Sampling ===
[TensorRT] ... (build messages)
Sampled indices: 2 0 4 4 
Test 1 passed!

=== Test 2: Statistical Distribution Check ===
[TensorRT] ... (build messages)
Index | Expected | Observed | Difference
------|----------|----------|------------
    0 |   0.1000 |   0.0987 |     0.0013 ✓
    1 |   0.2000 |   0.2023 |     0.0023 ✓
    2 |   0.3000 |   0.2965 |     0.0035 ✓
    3 |   0.2500 |   0.2534 |     0.0034 ✓
    4 |   0.1500 |   0.1491 |     0.0009 ✓
Test 2 passed!

=== Test 3: Large Batch Size ===
[TensorRT] ... (build messages)
Successfully processed batch size 128 with vocab size 1024
Test 3 passed!

=================================================
Test Results:
  Test 1 (Basic Sampling): PASSED
  Test 2 (Statistical Distribution): PASSED
  Test 3 (Large Batch): PASSED

All tests passed successfully!
```

## Key Implementation Details

### Plugin Interface

The test uses `IPluginV3` interface as recommended by TensorRT 10.x documentation:

```cpp
// Get plugin creator
auto* creator = getPluginRegistry()->getPluginCreator(
    "CategoricalSampling", "1", "");

// Create plugin
IPluginV3* pluginObj = creator->createPlugin(
    "CategoricalSampling", &emptyFC, TensorRTPhase::kBUILD);

// Add to network
IPluginV3Layer* pluginLayer = network->addPluginV3(
    inputs.data(), inputs.size(), nullptr, 0, *pluginObj);
```

### Memory Management

The test follows TensorRT best practices for resource management:
- Uses RAII with custom deleters for TensorRT objects
- Properly manages CUDA device memory
- Cleans up plugin objects after adding to network
- Uses smart pointers to prevent memory leaks

### Data Types

- **Input Probabilities**: FP16 (half precision) for efficiency
- **Top-k Indices**: INT32
- **Output Indices**: INT32

### Batch Processing

The plugin processes multiple samples in parallel:
- Each batch element gets independent random sampling
- Uses clock-based seeding for non-reproducible results
- Efficient GPU parallelization

## Troubleshooting

### TensorRT Not Found

If you see "TensorRT not found" errors:

```bash
# Set TensorRT path explicitly
export TENSORRT_ROOT=/path/to/tensorrt

# Or modify CMakeLists_test.txt to add your path
```

### CUDA Architecture Mismatch

If you get "no kernel image is available" errors:

1. Check your GPU compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

2. Edit `CMakeLists_test.txt` and adjust CUDA_ARCHITECTURES:
```cmake
set_property(TARGET CategoricalSamplingPlugin 
    PROPERTY CUDA_ARCHITECTURES 70 75 80 86 89 90)
```

Common architectures:
- 70: Tesla V100
- 75: RTX 2080, Titan RTX
- 80: A100
- 86: RTX 3090, RTX 3080
- 89: RTX 4090, RTX 4080
- 90: H100

### Statistical Test Variations

Test 2 may occasionally show warnings if statistical variation exceeds 5%:
- This is expected behavior due to randomness
- Re-run the test if needed
- Increase `numSamples` in the code for more stable results

### Plugin Registration Errors

If you see "Failed to get plugin creator":
- Ensure the plugin library is built correctly
- Check that `initLibNvInferPlugins()` succeeds
- Verify plugin name and version match: "CategoricalSampling" v"1"

## References

### TensorRT Documentation
- [Extending TensorRT with Custom Layers](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html)
- [IPluginV3 Interface](https://docs.nvidia.com/deeplearning/tensorrt/api/c_api/classnvinfer1_1_1_i_plugin_v3.html)
- [TensorRT Best Practices](https://docs.nvidia.com/deeplearning/tensorrt/best-practices/)

### Key TensorRT Concepts
- **IPluginV3**: Modern plugin interface (TensorRT 10.0+)
- **IPluginV3OneCore**: Core plugin capabilities
- **IPluginV3OneBuild**: Build-time plugin interface
- **IPluginV3OneRuntime**: Runtime plugin interface
- **Explicit Batch Mode**: Required for IPluginV3 plugins

## Architecture Notes

### Why IPluginV3?

The test uses `IPluginV3` because:
1. It's the recommended interface for TensorRT 10.0+
2. `IPluginV2` interfaces are deprecated
3. Provides better separation of build/runtime concerns
4. Supports modern TensorRT features

### Plugin Lifecycle

1. **Build Time**:
   - Plugin creator instantiates plugin
   - Plugin added to network
   - Engine built and serialized
   - Plugin object deleted

2. **Runtime**:
   - Engine deserialized from memory
   - Plugin recreated from serialized data
   - Execution context created
   - Inference performed

### Performance Considerations

- Uses FP16 for probability inputs (2x memory efficiency)
- Clock-based random seeding (no cuRAND state overhead)
- Parallel batch processing on GPU
- Direct device-to-device memory operations

## License

```
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: Apache-2.0
```
