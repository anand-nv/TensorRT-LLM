# CMakeLists.txt for categorical_sampling_plugin - includes kernel sources
# directly

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(CategoricalSamplingPlugin CUDA CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# Find TensorRT
find_path(
  TENSORRT_INCLUDE_DIR NvInfer.h
  HINTS ${TENSORRT_ROOT} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES include)

find_library(
  TENSORRT_LIBRARY_INFER nvinfer
  HINTS ${TENSORRT_ROOT} ${TENSORRT_BUILD} ${CUDA_TOOLKIT_ROOT_DIR}
  PATH_SUFFIXES lib lib64 lib/x64)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY_INFER)
  message(
    FATAL_ERROR
      "Cannot find TensorRT. Please set TENSORRT_ROOT or ensure TensorRT is in CUDA toolkit."
  )
endif()

include_directories(${TENSORRT_INCLUDE_DIR})

# CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90) # 91 (Blackwell) doesn't seem to be
                                             # supported in this codebase

# Plugin sources - include kernel sources directly
add_library(
  categorical_sampling_plugin SHARED CategoricalSamplingPlugin.cpp
                                     cuda_kernel/src/categorical_sampling.cu)

target_include_directories(
  categorical_sampling_plugin
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/cuda_kernel/include)

target_link_libraries(
  categorical_sampling_plugin PRIVATE ${TENSORRT_LIBRARY_INFER}
                                      ${CUDA_LIBRARIES} curand)

# Define NDEBUG for non-debug builds
target_compile_definitions(categorical_sampling_plugin
                           PRIVATE $<$<NOT:$<CONFIG:Debug>>:NDEBUG>)

# Set plugin properties
set_target_properties(
  categorical_sampling_plugin PROPERTIES CUDA_SEPARABLE_COMPILATION ON
                                         POSITION_INDEPENDENT_CODE ON)

# Set CUDA compile options
target_compile_options(
  categorical_sampling_plugin
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr --use_fast_math
          >)

# Install rules
install(
  TARGETS categorical_sampling_plugin
  LIBRARY DESTINATION /code/tensorrt_llm/build
  ARCHIVE DESTINATION /code/tensorrt_llm/build)

install(FILES CategoricalSamplingPlugin.h DESTINATION /code/tensorrt_llm/build)

# Print configuration
message(STATUS "=== Build Configuration ===")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "TensorRT Include: ${TENSORRT_INCLUDE_DIR}")
message(STATUS "TensorRT Library: ${TENSORRT_LIBRARY_INFER}")
