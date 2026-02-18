# Categorical sampling with TensorRT plugin

## Overview
Since TensorRT does not support categorical sampling, we add that as a **TensorRT Plugin**.

To include the sampling operation in a PyTorch model that we will be tracing to ONNX and TensorRT, we can refer to the plugin by name in the PyTorch code being traced (example below). During tracing, the operation becomes part of the ONNX graph (just a symbolic name, no implementation needed at this point). When `trtexec` loads the ONNX graph and encounters the sampling operation it looks for the corresponding name among its plugins. If we have provided the plugin path to TensorRT, it finds the plugin, loads it and executes it when running the model.

There are few pieces to discuss:

* Categorical sampling CUDA Kernel: implements categorical sampling on the GPU.

* Categorcial sampling TensorRT plugin (C++): wraps the CUDA kernel as a TensorRT plugin, implementing the TensorRT plugin APIs:
`IPluginV3`, `IPluginV3OneCore`, `IPluginV3OneBuild` and `IPluginV3OneRuntime`.

* `linear_lt_autoregressive.ipynb`: Demonstrates of how start from PyTorch code that uses the `CategoricalSampling` operation and trace it to ONNX and then TensorRT. It has an example of an autoregressive loop running a dummy LT (embedding layer + linear) and sampling at each step, all traced into a single graph via PyTorch -> ONNX -> TensorRT.

## Directory structure
```
tensorrt_llm/  (top-level)
├── linear_lt_autoregressive.ipynb      # demo notebook showing how to trace a model with the custom sampling operation
└── categorical_sampling_plugin/
    ├── README_plugin.md                # This file
    ├── build_plugin.sh                 # Builds the plugin
    ├── build_kernel_test.sh            # (optional) Builds the kernel unit test.
    ├── CMakeLists.txt                  # For building the plugin
    ├── CategoricalSamplingPlugin.cpp   # Plugin implementation
    ├── CategoricalSamplingPlugin.h     # Plugin header
    └── cuda_kernel/                    # Kernel subdirectory
        ├── CMakeLists.txt              # Kernel unit test build
        ├── include/
        │   └── categorical_sampling.cuh          # Kernel header
        ├── src/
        │   └── categorical_sampling.cu           # Kernel implementation
        └── test/
            └── test_categorical_sampling_fp16.cu # Kernel unit test
```
## How to build and use the plugin

### 1. Build the plugin

```bash
cd categorical_sampling_plugin
./build_plugin.sh
```
This will build the plugin (including the CUDA kernel) and copy the resulting shared library to:
`tensorrt_llm/build/libcategorical_sampling_plugin.so`.

Note that there is also a second script (`build_kernel_test.sh`) that builds a standalone unit test for the CUDA kernel (you don't need this just to use the plugin).

### 2. Define Categorical Sampling as a custom operation in your PyTorch code

```python
import torch
from torch import Tensor

class CategoricalSamplingFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor):
        # Forward is not used during ONNX parsing. So you can put a dummy implementation here.
        # We return a 1D INT32 tensor to match plugin output type.
        return torch.zeros(x.shape[0], dtype=torch.int32, device=x.device)

    @staticmethod
    def symbolic(g, x):
        # Emit ONNX node whose (domain, op_type) matches the TRT plugin creator.
        output = g.op("CategoricalSampling", x)
        # Set the output type to INT32 with 1D shape (dynamic size)
        output.setType(x.type().with_dtype(torch.int32).with_sizes([None]))
        return output

# IMPORTANT: The following function is what you call in your model. See `linear_lt_autoregressive.ipynb` for an example.
def categorical_sampling(x: Tensor) -> Tensor:
    return CategoricalSamplingFn.apply(x)
```

### 3. Trace the model to ONNX
Use `torch.export`. See `linear_lt_autoregressive.ipynb` for an example.

### 4. Convert to TensorRT and run
```bash
/usr/local/tensorrt/targets/x86_64-linux-gnu/bin/trtexec --fp16 \
    --onnx=/code/tensorrt_llm/local_transformer.onnx \
    --saveEngine=/code/tensorrt_llm/local_transformer.trt \
    --minShapes=hidden_states:2x768  \
    --optShapes=hidden_states:8x768 \
    --maxShapes=hidden_states:32x768 \
    --plugins=/code/tensorrt_llm/build/libcategorical_sampling_plugin.so
```
The last line tells TensorRT where to find our plugin.

# Notes
The plugin is currently built as a shared library (`*.so`). `trtexec` seems to load the library dynamically during execution but does **not** serialize it into the engine itself. We will need to figure out how this loading will work in the context of the TRT-LLM runtime. Options are: (1) load dynamically in TRT-LLM runtime, (2) statically link it into the TRT-LLM runtime, (3) serialize it into the engine itself (`trtexec` has an option `--setPluginsToSerialize` which seems related).
