/**
 * dropc function fprop/bprop host implementation
 */

#include <cassert>
#include <cstdlib>


//----------------------------------------------
//      host code declear: normal maskW
//----------------------------------------------
void computeFCDropC_fprop_h( 
      const float*  x,          ///<[in]  input matrix x, col major, numData x inDim
      const float*  w,          ///<[in]  weight matrix w, col major, inDim x outDim
      const float*  b,          ///<[in]  bias matrix, row major, 1 x outDim
      int outDim,               ///<[in]  output dimension
      int inDim,                ///<[in]  input dimension
      int numData,              ///<[in]  number of data in this batch
      const float * mw,         ///<[in]  maskWeights, col major, inDim x (outDimxnumData)
      const float * mb,         ///<[in]  maskBiases, col major, numData x outDim          
      float * y                 ///<[in,out] target matrix y, col major, numData x outDim
      ) {
   size_t m = outDim;
   size_t n = inDim;
   size_t d = numData;
   for( size_t j = 0; j < m; j++ ) {
      for( size_t i = 0; i < d; i++ ) {
         float c = 0;
         for( size_t k = 0; k < n; k++ ) {
            float x_ik = x[k*d+i];
            float w_kj = w[j*n+k];
            float mw_i_kj = mw[ i*m*n + j*n  + k ];
            // data: i, row: k, col, j
            assert( mw_i_kj == 0 || mw_i_kj == 1);
            c += x_ik * w_kj * mw_i_kj;
         }
         // inc bias
         float bj = b[j];
         float mb_i_j = mb[ j * d + i];
         assert( mb_i_j == 0 || mb_i_j == 1);
         c += bj * mb_i_j;
         // yij
         y[ j * d + i ] += c;
      }
   }
}

void computeFCDropC_bpropActs_h(
      const float*  v,         ///<[in]  bprop act from previous layer, col major,numData x outDim
      const float*  w,         ///<[in]  weight matrix w, col major, inDim x outDim
      int outDim,              ///<[in]  output dimension
      int inDim,               ///<[in]  input dimension
      int numData,             ///<[in]  number of data in this batch
      float scale_g,           ///<[in]  input gradient scale
      const float* mw,         ///<[in]  maskWeights, col major, inDim x (outDimxnumData)
      float* da,               ///<[in,out] d-active, col major, numData x inDim              
      float scale_da           ///<[in]  da scale
      ){
   size_t m = outDim;
   size_t n = inDim;
   size_t d = numData;
   for( size_t j = 0; j < n; j++ ) {
      for( size_t i = 0; i < d; i++ ) {
         float da_ij = da[ j * d + i];
         float c = 0;
         for( size_t k = 0; k < m; k++ ) {
            float v_ik = v[k*d+i];
            // wt_kj = w_jk
            float wt_kj = w[k*n+j];
            // data:i, row:j, col: k
            float mw_i_jk = mw[ i*m*n + k*n + j ];
            assert( mw_i_jk == 0 || mw_i_jk == 1);
            c += v_ik * wt_kj * mw_i_jk;
         }
         da[ j*d+i ] = scale_g * c + scale_da * da_ij;
      }
   }
}


void computeFCDropC_bpropWeights_h(
      const float* a,            ///<[in] prev activation matrix, col major, numData x inDim
      const float* v,            ///<[in] gradient matrix, col major, numData x outDim
      int outDim,                ///<[in]  output dimension              
      int inDim,                 ///<[in]  input dimension
      int numData,               ///<[in]  number of data in this batch
      float scale_g,             ///<[in] inc scale
      const float* mw,           ///<[in] maskWeights, col major, inDim x (outDimxnumData)
      float* dw,                 ///<[in,out] w gradient, col major, inDim x outDim
      float scale_dw             ///<[in] gradient scale
      ){
   size_t m = outDim;
   size_t n = inDim;
   size_t d = numData;
   for( size_t j = 0; j < m; j++ )
      for( size_t i = 0; i < n; i++ ) {
         float dw_ij = dw[j*n+i];
         float c = 0;
         for( size_t k = 0; k < d; k++ ) {
            // at_ik = a_ki
            float at_ik = a[i*d+k];
            float v_kj = v[j*d+k];
            // data:k row:i col:j
            float mw_k_ij = mw[ k*m*n + j*n + i];
            assert( mw_k_ij == 0 || mw_k_ij == 1);
            c+= at_ik * v_kj * mw_k_ij;
         }
         dw[j*n+i] = scale_g * c + scale_dw*dw_ij;
      }
}

