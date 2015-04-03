/**
 * cuda debug helper file
 *  provides a set of function to print a chunk of device memory into file for inspection
 */

#ifndef __CUDA_DEBUG_UTILS_HPP__
#define __CUDA_DEBUG_UTILS_HPP__ 

#include <string>
#include <vector>

#include "cuda_common.hpp"
#include "debug_utils.hpp"

/** print 2d device memory to file */
template<class T>
void printDeviceMemory( 
      const T* m_dev,              ///< [in] device memory
      size_t width,                ///< [in] device memory width
      size_t height,               ///< [in] device memory height
      size_t depth,                ///< [in] device memory depth
      size_t heightStride,         ///< [in] device memory height stride
      const std::string& fileName  ///< [in] output file 
      ){
   // create & copy to host memory
   std::vector<T> m( width * height * depth);
   checkCuda( cudaMemcpy2D( 
            &m[0], width * sizeof(T),
            m_dev, heightStride * sizeof(T),
            width * sizeof(T), height * depth, cudaMemcpyDeviceToHost 
            ) );

   printHostMemory( &m[0], width, height, depth, fileName );
}

#endif
