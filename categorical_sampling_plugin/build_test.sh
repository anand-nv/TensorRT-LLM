#!/bin/bash

# Script to build and run the CategoricalSamplingPlugin test

set -e  # Exit on error

echo "=========================================="
echo "Building CategoricalSamplingPlugin Test"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory
BUILD_DIR="$SCRIPT_DIR/build_test"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Check for TensorRT
TENSORRT_PATHS=(
    "/usr/local/tensorrt"
    "/usr/local/TensorRT"
    "/opt/tensorrt"
    "$HOME/TensorRT"
)

TENSORRT_ROOT=""
for path in "${TENSORRT_PATHS[@]}"; do
    if [ -d "$path" ]; then
        TENSORRT_ROOT="$path"
        echo "Found TensorRT at: $TENSORRT_ROOT"
        break
    fi
done

if [ -z "$TENSORRT_ROOT" ]; then
    echo "WARNING: TensorRT not found in default locations."
    echo "Please set TENSORRT_ROOT environment variable or adjust the script."
    echo "Trying to build anyway..."
fi

# Copy CMakeLists.txt for test build
cp "$SCRIPT_DIR/CMakeLists_test.txt" "$BUILD_DIR/CMakeLists.txt"

# Configure with CMake
echo ""
echo "Configuring with CMake..."
if [ -n "$TENSORRT_ROOT" ]; then
    cmake -DTENSORRT_ROOT="$TENSORRT_ROOT" \
          -DCMAKE_BUILD_TYPE=Release \
          "$SCRIPT_DIR"
else
    cmake -DCMAKE_BUILD_TYPE=Release \
          "$SCRIPT_DIR"
fi

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "To run the test:"
echo "  cd $BUILD_DIR"
echo "  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$BUILD_DIR/lib"
echo "  ./bin/test_plugin"
echo ""

