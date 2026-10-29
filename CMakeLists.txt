cmake_minimum_required(VERSION 3.22)
project(distributed_mm_benchmark LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP)

find_path(NCCL_INCLUDE_DIR NAMES nccl.h HINTS ENV NCCL_HOME PATH_SUFFIXES include REQUIRED)
find_library(NCCL_LIBRARY NAMES nccl HINTS ENV NCCL_HOME PATH_SUFFIXES lib lib64 REQUIRED)
find_path(NVSHMEM_INCLUDE_DIR NAMES nvshmem.h HINTS ENV NVSHMEM_HOME PATH_SUFFIXES include REQUIRED)

# NVSHMEM releases differ: some provide libnvshmem.so; newer installations
# can provide libnvshmem_host.so plus libnvshmem_device.a.
find_library(NVSHMEM_LIBRARY NAMES nvshmem nvshmem_host
  HINTS ENV NVSHMEM_HOME PATH_SUFFIXES lib lib64 REQUIRED)
# Split NVSHMEM distributions require host and device libraries.
# Legacy monolithic libnvshmem.a must NOT be combined with nvshmem_device.
set(NVSHMEM_LINK_LIBRARIES "${NVSHMEM_LIBRARY}")
if(NVSHMEM_LIBRARY MATCHES "nvshmem_host")
  find_library(NVSHMEM_DEVICE_LIBRARY NAMES nvshmem_device
    HINTS ENV NVSHMEM_HOME PATH_SUFFIXES lib lib64 REQUIRED)
  list(APPEND NVSHMEM_LINK_LIBRARIES "${NVSHMEM_DEVICE_LIBRARY}")
  message(STATUS "Using split NVSHMEM: ${NVSHMEM_LIBRARY} + ${NVSHMEM_DEVICE_LIBRARY}")
else()
  message(STATUS "Using monolithic NVSHMEM: ${NVSHMEM_LIBRARY}")
endif()
find_library(NVML_LIBRARY NAMES nvidia-ml HINTS /usr/lib64 /usr/lib/x86_64-linux-gnu)

foreach(variant IN ITEMS 1D-row-block 2D-block-matrix)
  if(variant STREQUAL "1D-row-block")
    set(exe mm_1d)
  else()
    set(exe mm_2d)
  endif()
  add_executable(${exe} ${variant}/mmb.cu ${variant}/mulmat_kernel.cu)
  target_include_directories(${exe} PRIVATE ${NCCL_INCLUDE_DIR} ${NVSHMEM_INCLUDE_DIR})
  target_link_libraries(${exe} PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cuda_driver
    ${NCCL_LIBRARY} ${NVSHMEM_LINK_LIBRARIES})
  # CUDA separable compilation performs device linking for split NVSHMEM.

  if(NVML_LIBRARY)
    target_link_libraries(${exe} PRIVATE ${NVML_LIBRARY})
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${exe} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${exe} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
  endif()
  set_target_properties(${exe} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
endforeach()


