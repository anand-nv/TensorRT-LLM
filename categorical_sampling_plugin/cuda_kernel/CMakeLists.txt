# CMakeLists.txt for cuda_kernel - Standalone build with tests only Build plugin
# from parent directory to include kernel sources directly

cmake_minimum_required(VERSION 3.18)
project(categorical_sampling_kernel LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# CUDA architecture (adjust based on your GPU)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)
endif()

# Source files
set(CUDA_KERNEL_SOURCES src/categorical_sampling.cu)

# Create library (for tests to link against)
add_library(categorical_sampling_kernel STATIC ${CUDA_KERNEL_SOURCES})

# Set CUDA flags
target_compile_options(
  categorical_sampling_kernel
  PRIVATE $<$<COMPILE_LANGUAGE:CUDA>: --expt-relaxed-constexpr --use_fast_math
          -Xcompiler=-Wall >)

# Test executables
add_executable(test_categorical_sampling_fp16
               test/test_categorical_sampling_fp16.cu)
target_link_libraries(test_categorical_sampling_fp16
                      categorical_sampling_kernel ${CUDA_LIBRARIES} curand)

# Enable testing
enable_testing()
add_test(NAME CategoricalSamplingFP16Test
         COMMAND test_categorical_sampling_fp16)

# Print configuration
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "CUDA Include Dirs: ${CUDA_INCLUDE_DIRS}")
