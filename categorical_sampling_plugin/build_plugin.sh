#!/bin/bash
# Builds the plugin (includes kernel sources directly)

# Parse command line options. Include a "-d" flag to build in Debug mode.
BUILD_TYPE="Release"
while getopts "d" opt; do
  case $opt in
    d)
      BUILD_TYPE="Debug"
      ;;
    \?)
      echo "Usage: $0 [-d]"
      echo "  -d: Build in Debug mode (default is Release)"
      exit 1
      ;;
  esac
done

set -e

cd "$(dirname "$0")"
mkdir -p build
cd build
cmake -DTENSORRT_ROOT=/usr/local/tensorrt -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..
# clean each time (building is fast)
make clean
echo make -j$(nproc)
make install

echo ""
echo "Plugin installed to /code/tensorrt_llm/build/"
ls -lh /code/tensorrt_llm/build/libcategorical_sampling_plugin.so
