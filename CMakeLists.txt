cmake_minimum_required(VERSION 3.24)

# ============================================================
# MGXBench - Multi-GPU Execution Benchmark
# CUDA / MPI / NCCL / NVSHMEM
#
# Supported architectures:
#   NVIDIA V100       : sm_70 (CUDA 11.x/12.x)
#   NVIDIA H100/GH200 : sm_90 (CUDA 13.x)
# ============================================================

# CUDA architecture can be overridden externally.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(MGXBench LANGUAGES CXX CUDA)

# ============================================================
# C++ and CUDA standards
# ============================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

# ============================================================
# Dependencies
# ============================================================

find_package(CUDAToolkit REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

# NCCL
find_path(NCCL_INCLUDE_DIR
    NAMES nccl.h
    HINTS ENV NCCL_HOME ENV NCCL_ROOT
    PATH_SUFFIXES include
    REQUIRED
)

find_library(NCCL_LIBRARY
    NAMES nccl
    HINTS ENV NCCL_HOME ENV NCCL_ROOT
    PATH_SUFFIXES lib lib64
    REQUIRED
)

# NVSHMEM include directory
find_path(NVSHMEM_INCLUDE_DIR
    NAMES nvshmem.h
    HINTS ENV NVSHMEM_HOME ENV NVSHMEM_ROOT
    PATH_SUFFIXES include
    REQUIRED
)

# NVSHMEM host library
find_library(NVSHMEM_HOST_LIBRARY
    NAMES nvshmem_host
    HINTS ENV NVSHMEM_HOME ENV NVSHMEM_ROOT
    PATH_SUFFIXES lib lib64
    REQUIRED
)

# NVSHMEM device library
find_library(NVSHMEM_DEVICE_LIBRARY
    NAMES nvshmem_device
    HINTS ENV NVSHMEM_HOME ENV NVSHMEM_ROOT
    PATH_SUFFIXES lib lib64
    REQUIRED
)

# ============================================================
# Shared dependencies
# ============================================================

add_library(mgx_dependencies INTERFACE)

target_include_directories(mgx_dependencies INTERFACE
    ${NCCL_INCLUDE_DIR}
    ${NVSHMEM_INCLUDE_DIR}
)

target_link_libraries(mgx_dependencies INTERFACE
    MPI::MPI_CXX
    CUDA::cudart
    ${NCCL_LIBRARY}
    ${NVSHMEM_HOST_LIBRARY}
    ${NVSHMEM_DEVICE_LIBRARY}
)

# ============================================================
# 1D Row-Block Matrix Multiplication
# ============================================================

add_executable(mm_1d
    1D-row-block/mmb.cu
    1D-row-block/mulmat_kernel.cu
)

target_link_libraries(mm_1d PRIVATE
    mgx_dependencies
)

# ============================================================
# 2D SUMMA Matrix Multiplication
# ============================================================

add_executable(mm_2d
    2D-block-matrix/mmb.cu
    2D-block-matrix/mulmat_kernel.cu
)

target_link_libraries(mm_2d PRIVATE
    mgx_dependencies
)

# ============================================================
# CUDA device linking
# ============================================================

set_target_properties(mm_1d mm_2d PROPERTIES
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# ============================================================
# Configuration summary
# ============================================================

message(STATUS "========================================")
message(STATUS "MGXBench Build Configuration")
message(STATUS "========================================")
message(STATUS "CUDA Architecture : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "C++ Standard      : ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA C++ Standard : ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Toolkit      : ${CUDAToolkit_VERSION}")
message(STATUS "MPI CXX Compiler  : ${MPI_CXX_COMPILER}")
message(STATUS "NCCL Library      : ${NCCL_LIBRARY}")
message(STATUS "NVSHMEM Host      : ${NVSHMEM_HOST_LIBRARY}")
message(STATUS "NVSHMEM Device    : ${NVSHMEM_DEVICE_LIBRARY}")
message(STATUS "========================================")


