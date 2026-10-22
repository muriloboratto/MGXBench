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
# libnvshmem.a already contains CUDA device symbols. Do not link
# libnvshmem_device.a together with the monolithic archive.
# NVSHMEM_DEVICE_LIBRARY is intentionally not used with libnvshmem.a.
find_library(NVML_LIBRARY NAMES nvidia-ml HINTS /usr/lib64 /usr/lib/x86_64-linux-gnu)

message(STATUS "NVSHMEM host/legacy library: ${NVSHMEM_LIBRARY}")
message(STATUS "NVSHMEM device library: not linked separately")
if(NVSHMEM_LIBRARY MATCHES "nvshmem_host")
  message(FATAL_ERROR "This build expects monolithic libnvshmem.a. Configure with -DNVSHMEM_LIBRARY=$ENV{NVSHMEM_HOME}/lib/libnvshmem.a")
endif()

foreach(variant IN ITEMS 1D-row-block 2D-block-matrix)
  if(variant STREQUAL "1D-row-block")
    set(exe mm_1d)
  else()
    set(exe mm_2d)
  endif()
  add_executable(${exe} ${variant}/mmb.cu ${variant}/mulmat_kernel.cu)
  target_include_directories(${exe} PRIVATE ${NCCL_INCLUDE_DIR} ${NVSHMEM_INCLUDE_DIR})
  target_link_libraries(${exe} PRIVATE MPI::MPI_CXX CUDA::cudart CUDA::cuda_driver
    ${NCCL_LIBRARY} ${NVSHMEM_LIBRARY})
  # Do not link libnvshmem_device.a alongside legacy libnvshmem.a:
  # both contain CUDA device symbols and cause nvlink multiple definitions.

  if(NVML_LIBRARY)
    target_link_libraries(${exe} PRIVATE ${NVML_LIBRARY})
  endif()
  if(OpenMP_CXX_FOUND)
    target_link_libraries(${exe} PRIVATE OpenMP::OpenMP_CXX)
    target_compile_options(${exe} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fopenmp>)
  endif()
  set_target_properties(${exe} PROPERTIES CUDA_SEPARABLE_COMPILATION ON CUDA_STANDARD 14)
endforeach()

