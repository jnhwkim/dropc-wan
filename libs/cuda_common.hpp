/**
 * cuda common header file
 *
 */

#ifndef __CUDA_COMMON_H__
#define __CUDA_COMMON_H__

#include <iostream>
#include <driver_types.h>
#include <cuda_runtime.h>
#include <cuda.h>

#define checkCuda(x) checkCuda_function( (x), __FILE__, __LINE__ )
#define checkLastCudaError() checkLastCudaError_function( __FILE__, __LINE__ )

//-----------------------------------------
//         cuda utility function
//-----------------------------------------
inline void checkCuda_function( cudaError_t e, const char* file, const int line ) {
   if( e != cudaSuccess ) {
      std::cerr << file << "(" << line << ")";
      std::cerr <<  "CUDA Error: " <<  cudaGetErrorString( e ) << std::endl;
      exit(-1);
   }
}

inline void checkCuda_function( cudaError_enum e, const char* file, const int line ) {
   if( e != CUDA_SUCCESS ) {
      std::cerr << file << "(" << line << ")";
      std::cerr <<  "CUDA Error "  << std::endl;
      exit(-1);
   }
}

inline void checkLastCudaError_function( const char* file, const int line) {
    checkCuda_function(cudaGetLastError(), file, line );
}

inline int divup( int x, int y ) {
   return (x + y - 1)/y;
}

#endif
