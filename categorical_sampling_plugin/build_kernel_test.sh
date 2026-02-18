#!/bin/bash
# Build CUDA kernel standalone with tests

set -e

cd "$(dirname "$0")/cuda_kernel"
mkdir -p build
cd build
cmake ..
make -j$(nproc)

echo ""
echo "Build complete! Run test with:"
echo "  ./cuda_kernel/build/test_categorical_sampling_fp16"
