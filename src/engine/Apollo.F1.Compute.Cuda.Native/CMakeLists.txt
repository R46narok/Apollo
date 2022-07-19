cmake_minimum_required(VERSION 3.21)
project(Apollo.F1.Compute.Cuda.Native CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_library(Apollo.F1.Compute.Cuda.Native SHARED src/pointwise_operations.cu src/pointwise_operations.cuh src/memory.cu src/memory.cuh src/functions_operations.cu src/functions_operations.cuh src/transpose.cu src/transpose.cuh src/multiplication.cu src/multiplication.cuh src/transformations.cu src/transformations.cuh src/scalar.cu src/scalar.cuh src/sum.cu src/sum.cuh src/core.cuh)

set_target_properties(Apollo.F1.Compute.Cuda.Native PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_include_directories(Apollo.F1.Compute.Cuda.Native PRIVATE src)
target_compile_options(Apollo.F1.Compute.Cuda.Native PRIVATE -arch=sm_60)
if(WIN32)
    # copy the .dll file to the same folder as the executable
    add_custom_command(
            TARGET ${PROJECT_NAME}
            POST_BUILD
            COMMAND
            powershell -File ${CMAKE_CURRENT_LIST_DIR}/scripts/CopyOutputDll.ps1
            VERBATIM)
endif()