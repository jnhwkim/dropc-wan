cmake_minimum_required( VERSION 2.6 )
project (dropc-kernel)

# find package
find_package( CUDA )

include( FindCUDA )

# debug
if( "${CMAKE_BUILD_TYPE}" STREQUAL "Debug" )
   set( CUDA_NVCC_FLAGS "-arch=sm_20" "-G" "-g" )
   set( CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas" )
   #set( CMAKE_CXX_FLAGS "-Wall" )
   message( "-------------------debug---------------------" )
# release O1
elseif( "${CMAKE_BUILD_TYPE}" STREQUAL "ReleaseO1" )
   set( CUDA_NVCC_FLAGS "-arch=sm_20" "-UNDEBUG" ) 
   SET(CMAKE_CXX_FLAGS_DISTRIBUTION "-O1")
   set( CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas" )
   #set( CMAKE_CXX_FLAGS "-Wall" )
   #enable assert
   add_definitions( -UNDEBUG )
   message( "----------------release: O1--------------------" )
# release
elseif( "${CMAKE_BUILD_TYPE}" STREQUAL "Release" )
   set( CUDA_NVCC_FLAGS "-arch=sm_20" "-UNDEBUG" "-lineinfo") 
   set( CMAKE_CXX_FLAGS "-Wall -Wno-unknown-pragmas" )
   #set( CMAKE_CXX_FLAGS "-Wall" )
   #enable assert
   add_definitions( -UNDEBUG )
   message( "--------------------release--------------------" )
endif( "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")

link_libraries( 
   "-ldl -lm -lpthread -lX11" 
   "-L/usr/lib/atlas-base/" 
   "-L/usr/lib64/atlas/" 
   "-lcblas" 
   "-L/home/wanli/NVIDIA_GPU_Computing_SDK/C/lib/-lcutil_x86_64" 
   )
include_directories( 
   "/home/wanli/NVIDIA_GPU_Computing_SDK/C/common/inc" 
   "/home/wanli/NVIDIA_GPU_Computing_SDK/shared/inc" 
   )

set( HOST_MATRIX
   libs/matrix.cpp
   )

set( DEV_MATRIX
   libs/nvmatrix.cu
   libs/nvmatrix_kernels.cu
   )

# test dropc kerneel
cuda_add_executable( test_dropc
   test_dropc.cu
   dropc_host.cpp
   dropc_dev.cu
   ${HOST_MATRIX}
   ${DEV_MATRIX}
   )
cuda_add_cublas_to_target( test_dropc )


# test dropc kerneel
cuda_add_executable( test_dropc_bit
   test_dropc_bit.cu
   dropc_bit_host.cpp
   dropc_bit_dev.cu
   ${HOST_MATRIX}
   ${DEV_MATRIX}
   )
cuda_add_cublas_to_target( test_dropc_bit )
