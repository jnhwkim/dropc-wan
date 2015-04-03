/**
 *  dropc function fprop/bprop host declear 
 *  All matrix ptr are col major
 */

#ifndef __DROPC_BIT_HOST_HPP__
#define __DROPC_BIT_HOST_HPP__

//----------------------------------------------
//       host code declear: bits maskW
//----------------------------------------------
void computeFCDropC_bit_fprop_h( 
      const float*  x,        ///<[in]  input matrix x, col major, numData x inDim
      const float*  w,        ///<[in]  weight matrix w, col major, inDim x outDim
      const float*  b,        ///<[in]  bias matrix, row major, 1 x outDim
      int outDim,             ///<[in]  output dimension
      int inDim,              ///<[in]  input dimension
      int numData,            ///<[in]  number of data in this batch
      const unsigned int* mw, ///<[in]  maskWeights,col major,inDimx(outDimxnumData/sizeof(uint))
      const float * mb,       ///<[in]  maskBiases, col major, numData x outDim          
      float * y               ///<[in,out] target matrix y, col major, numData x outDim
      );


void computeFCDropC_bit_bpropActs_h(
      const float*  v,        ///<[in]  bprop act from previous layer, col major,numData x outDim
      const float*  w,        ///<[in]  weight matrix w, col major, inDim x outDim
      int outDim,             ///<[in]  output dimension
      int inDim,              ///<[in]  input dimension
      int numData,            ///<[in]  number of data in this batch
      float scale_g,          ///<[in]  input gradient scale
      const unsigned int* mw, ///<[in]  maskWeights, col major,inDimx(outDimxnumData/sizeof(uint))
      float* da,              ///<[in,out] d-active, col major, numData x inDim              
      float scale_da          ///<[in]  da scale
      );

void computeFCDropC_bit_bpropWeights_h(
      const float* a,         ///<[in] prev activation matrix, col major, numData x inDim
      const float* v,         ///<[in] gradient matrix, col major, numData x outDim
      int outDim,             ///<[in]  output dimension              
      int inDim,              ///<[in]  input dimension
      int numData,            ///<[in]  number of data in this batch
      float scale_g,          ///<[in] inc scale
      const unsigned int* mw, ///<[in]  maskWeights, col major, inDim x (outDimxnumData/sizeof)
      float* dw,              ///<[in,out] w gradient, col major, inDim x outDim
      float scale_dw          ///<[in] gradient scale
      );

#endif
