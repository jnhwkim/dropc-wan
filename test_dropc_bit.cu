/**
 *  dropc kerenel testing code
 */

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include "libs/matrix.h"
#include "libs/nvmatrix.cuh"
#include "libs/cuda_common.hpp"

#include "dropc_bit_host.hpp"
#include "dropc_bit_dev.hpp"

//#define PRINT_MATRIX

using namespace std;

//---------------------------------------------------------------------
//   test util functions
//---------------------------------------------------------------------
Matrix rand_col_matrix( int r, int c ) {
   Matrix m( r, c );
   m.randomizeUniform();
   m.setTrans( true ); // col major matrix
   return m;
}

void print_matrix( const Matrix& m, const string& text ) {
   m.printShape( text.c_str() );
   cout << m.isTrans() << endl;
   m.print();
}

void print_matrix( const NVMatrix& m, const string& text ) {
   m.printShape( text.c_str() );
   cout << m.isTrans() << endl;
   int r = m.getNumRows();
   int c = m.getNumCols();
   m.print( r, c );
}

void compare_matrix( Matrix& lm, Matrix& rm, const string& text ) {
   assert( lm.getNumRows() == rm.getNumRows() );
   assert( lm.getNumCols() == rm.getNumCols() );
   assert( lm.isTrans() == rm.isTrans() );
   int r = lm.getNumRows();
   int c = lm.getNumCols();
   bool isTrans = lm.isTrans();
   Matrix diff = rand_col_matrix( r,c );
   diff.setTrans( isTrans );
   lm.subtract( rm, diff );
#ifdef PRINT_MATRIX
   print_matrix( diff, text );
#endif
   cout << text << endl;
   cout << "mean : " << diff.sum()/(r*c) << endl;
   cout << "max  : " << diff.max() << endl;
   cout << "min  : " << diff.min() << endl;
}

cudaEvent_t start, stop;
void startDevTimer() {
   checkCuda( cudaEventCreate( &start ) );
   checkCuda( cudaEventCreate( &stop ) );
   checkCuda( cudaEventRecord( start, 0 ) );
}

float stopDevTimer() {
   checkCuda( cudaEventRecord( stop, 0 ) );
   checkCuda( cudaEventSynchronize( stop ) );
   float elapsedTime;
   checkCuda( cudaEventElapsedTime( &elapsedTime, start, stop ) );
   checkCuda( cudaEventDestroy( start ) );
   checkCuda( cudaEventDestroy( stop ) );
   return elapsedTime;
}

//void random_mw( unsigned int* mw, int n ) {
//   //unsigned int max_value = 1 << (sizeof( unsigned int )*8);
//   for( int i = 0; i < n; i++ ) {
//      //mw[i] = static_cast<unsigned int>( rand() % max_value );
//      mw[i] = rand();
//   }
//}

void copyDevMaskWeights( 
      const MaskWeights& mw,
      vector<unsigned int>& mw_host
      ){
   size_t width = mw.get_width();
   size_t height = mw.get_height();
   mw_host.assign( width * height, 0 );
   checkCuda( cudaMemcpy2D(
            &mw_host[0], width * sizeof( unsigned int ),
            mw.data(), mw.stride() * sizeof( unsigned int ), 
            width * sizeof(unsigned int), height, cudaMemcpyDeviceToHost ) );
}

void print_mw( const vector<unsigned int>& mw, int width, int height ) {
   cout << "mw: " << height << "x" << width << endl;
   for( int j = 0; j < height; j++ ) {
      for( int i = 0; i < width; i++ ) {
         cout << mw.at( j * width + i ) << " "; 
      }
      cout << endl;
   }
}

void count_bits( const vector<unsigned int>& mw ) {
   int num_bits = 8 * sizeof(unsigned int);
   vector<int> hist( 2, 0 );
   for( int i = 0; i < (int)mw.size(); i++ ) {
      for( int j = 0; j < num_bits; j++ ) {
         unsigned int r = mw[i] & ( 1 << j );
         if( r == 0 )
            hist[0]++;
         else
            hist[1]++;
      }
   }
   // print count
   cout << "hist: ";
   copy( hist.begin(), hist.end(), ostream_iterator<int>( cout, " " ) );
   cout << " on ratio: " << static_cast<float>(hist[1])/( hist[0]+hist[1] );
   cout << endl;
}

//---------------------------------------------------------------------

void test_inference() {
    int d = 128;
    int m = 1024;
    int numSamples = 10000;

    NVMatrix mu( d, m );
    mu.apply( NVMatrixOps::Zero() );
    //mu.apply( NVMatrixOps::One() ); mu.scale(10);
    NVMatrix var( d, m );
    var.apply( NVMatrixOps::One() );

    NVMatrix y( d, m );
   //---------------
   // start gpu timer
   //---------------
   startDevTimer();
    computeFCDropC_bit_inference_d( 
            mu.getDevData(), var.getDevData(), 
            d*m,
            numSamples, 
            y.getDevData() );
   //---------------
   // stop gpu timer
   //---------------
   float elapsedTime_host = stopDevTimer();
   cout << "GPU timer time: " << elapsedTime_host << "ms" << endl;

    y.print( 0, 5, 0, 5 );
}

void test_maskweights (){
   //int m = 5;
   //int n = 9;
   //int d = 8;

   //int m = 1024;
   //int n = 1024;
   //int d = 128;

   int m = 900;
   int n = 807;
   int d = 110;

   vector<unsigned int> mw_host;
//-------------------------------------
   cout << "Default prob(1) = 0.5: " << endl;
   MaskWeights mw;
   mw.resize( m, n, d );
   mw.randomize();
   copyDevMaskWeights( mw, mw_host );
   count_bits( mw_host );

   mw.randomize();
   copyDevMaskWeights( mw, mw_host );
   count_bits( mw_host );

//-------------------------------------
   cout << "prob(1) = 0.4: " << endl;
   MaskWeights mw2( 0.4 );
   mw2.resize( m, n, d );
   mw2.randomize();
   //vector<unsigned int> mw_host;
   copyDevMaskWeights( mw2, mw_host );
   count_bits( mw_host );

   mw2.randomize();
   copyDevMaskWeights( mw2, mw_host );
   count_bits( mw_host );

//-------------------------------------
   cout << "prob(1) = 0.6: " << endl;
   MaskWeights mw3( 0.6 );
   mw3.resize( m, n, d );
   mw3.randomize();
   //vector<unsigned int> mw_host;
   copyDevMaskWeights( mw3, mw_host );
   count_bits( mw_host );

   //---------------
   // start gpu timer
   //---------------
   startDevTimer();
   // randomize
   mw3.randomize();
   //---------------
   // stop gpu timer
   //---------------
   float elapsedTime_host = stopDevTimer();

   copyDevMaskWeights( mw3, mw_host );
   count_bits( mw_host );
   cout << "Randomize GPU timer time: " << elapsedTime_host << "ms" << endl;
}

void test_fprop() {
   cout << " Drop Connection fprop: " << endl;
   //int m = 5;
   //int n = 8;
   //int d = 3;

   int m = 1024;
   int n = 1024;
   int d = 128;

   //int m = 900;
   //int n = 807;
   //int d = 110;
   
   //-----------------------------------------
   // set up maskWeight
   //-----------------------------------------
   MaskWeights mw;
   mw.resize( m, n, d );
   mw.randomize();
   vector<unsigned int> mw_host;
   copyDevMaskWeights( mw, mw_host );
#ifdef PRINT_MATRIX
   print_mw( mw_host, mw.get_width(), mw.get_height() );
#endif

   //-----------------------------------------
   //        host code
   //-----------------------------------------
   // input
   Matrix x = rand_col_matrix( d,n );
   //print_matrix( x, "x matrix" );
   Matrix w = rand_col_matrix( n, m );
   //print_matrix( w, "w matrix" );
   Matrix b = rand_col_matrix( 1, m );
   //print_matrix( b, "b matrix" );
   b.setTrans( false );

   Matrix mb = rand_col_matrix( d, m );
   mb.biggerThanScalar( 0.5 );
   //print_matrix( mb, "mb matrix" );

   // output
   Matrix y = rand_col_matrix( d, m );
   Matrix y_prev(y);
   y_prev.setTrans(true);
   y.copy( y_prev );

   //print_matrix( y, "y before host compute" );

   //---------------
   // start gpu timer
   //---------------
   startDevTimer();
   // call fprop
   computeFCDropC_bit_fprop_h( 
         x.getData(), w.getData(), b.getData(), // input matrix
         m, n, d, // dims
         &mw_host[0],
         mb.getData(),  // masks
         y.getData()        // output
         );
   //---------------
   // stop gpu timer
   //---------------
   float elapsedTime_host = stopDevTimer();
   cout << "GPU timer time: " << elapsedTime_host << "ms" << endl;

   // print y
#ifdef PRINT_MATRIX
   print_matrix( y, "y after host compute" );
#endif

   //-----------------------------------------
   //        dev code
   //-----------------------------------------
   // input
   NVMatrix x_dev( d, n, true );
   x_dev.copyFromHost( x );

   NVMatrix w_dev( n, m, true );
   w_dev.copyFromHost( w );

   NVMatrix b_dev( 1, m, false );
   b_dev.copyFromHost( b );

   // masks
   NVMatrix mb_dev( d, m, true );
   mb_dev.copyFromHost( mb );

   // output
   NVMatrix y_dev( d, m, true );
   y_dev.copyFromHost( y_prev );

#ifdef PRINT_MATRIX
   //print_matrix( y_dev, "y_dev before dev compute" );
#endif
   //---------------
   // start gpu timer
   //---------------
   startDevTimer();

   //--------------
   // call fprop
   //--------------
   computeFCDropC_bit_fprop_d(
         x_dev.getDevData(), w_dev.getDevData(), b_dev.getDevData(), // input matrix
         m, n, d, // dims
         mw,  //mask w
         mb_dev.getDevData(),  // mask b
         y_dev.getDevData()        // output
         );
   //---------------
   // stop gpu timer
   //---------------
   float elapsedTime_dev = stopDevTimer();
   cout << "GPU timer time: " << elapsedTime_dev << "ms" << endl;

   // host output
   //print_matrix( y_dev, "y_dev after dev compute" );
   //y_dev.print( 0, d, 0, m );
   Matrix y2 = rand_col_matrix( d, m );
   y_dev.copyToHost( y2 );
#ifdef PRINT_MATRIX
   print_matrix( y2, "y2 after dev compute" );
#endif

   // diff
   compare_matrix( y, y2, "y-y2" );
   // print speed up
   cout << "speed up: " << elapsedTime_host/elapsedTime_dev << endl;

   //-----------------------------------------
   //    dev code: lower bound 
   //    cublas + copy time/2
   //-----------------------------------------
   // compute wx+b
   startDevTimer();
   y_dev.addProduct( x_dev, w_dev, 0, 1 );
   y_dev.addVector( b_dev );
   float elapsedTime_cublas = stopDevTimer();

   // copy data: read/write
   MaskWeights mw2( 0.5 );
   mw2.resize( m, n, d );
   startDevTimer();
   checkCuda( cudaMemcpy2D( 
       const_cast<unsigned int*>( mw2.data() ), 
       mw2.stride() * sizeof( unsigned int ),
       mw.data(),
       mw.stride() * sizeof( unsigned int ),
       mw.get_width() * sizeof( unsigned int ),
       mw.get_height(), cudaMemcpyDeviceToDevice ) );
   // divide 2 because copy includes read and write
   elapsedTime_cublas += stopDevTimer()/2; 
   cout << "Lower Bound(cublas+read memory): " << 
       elapsedTime_cublas << "ms" << endl;
}

void test_bpropa() {
   cout << " Drop Connection bprop acts: " << endl;
   //int m = 5;
   //int n = 8;
   //int d = 3;

   int m = 1024;
   int n = 1024;
   int d = 128;

   //int m = 900;
   //int n = 807;
   //int d = 110;

   //-----------------------------------------
   // set up maskWeight
   //-----------------------------------------
   MaskWeights mw;
   mw.resize( m, n, d );
   mw.randomize();
   vector<unsigned int> mw_host;
   copyDevMaskWeights( mw, mw_host );
#ifdef PRINT_MATRIX
   print_mw( mw_host, mw.get_width(), mw.get_height() );
#endif

   //-----------------------------------------
   //        host code
   //-----------------------------------------
   // input
   Matrix v = rand_col_matrix( d,m );
   Matrix w = rand_col_matrix( n,m );

   // output
   Matrix da = rand_col_matrix(d,n);
   Matrix da_prev( da ); 
   da_prev.setTrans(true);
   da.copy( da_prev );

   //---------------
   // start gpu timer
   //---------------
   startDevTimer();
   // call bpropa
   computeFCDropC_bit_bpropActs_h(
         v.getData(), w.getData(),
         m, n, d,
         1,
         &mw_host[0],
         da.getData(), 1
         );
   //---------------
   // stop gpu timer
   //---------------
   float elapsedTime_host = stopDevTimer();
   cout << "GPU timer time: " << elapsedTime_host << "ms" << endl;

   // print output
#ifdef PRINT_MATRIX
   print_matrix( da, "da after host compute" );
#endif

   //-----------------------------------------
   //        dev code
   //-----------------------------------------
   // input
   NVMatrix v_dev( d, m, true );
   v_dev.copyFromHost( v );

   NVMatrix w_dev( n, m, true );
   w_dev.copyFromHost( w );

   // output
   NVMatrix da_dev( d, n, true );
   da_dev.copyFromHost( da_prev );

   //---------------
   // start gpu timer
   //---------------
   startDevTimer();

   //---------------
   // call bpropa
   //---------------
   computeFCDropC_bit_bpropActs_d(
         v_dev.getDevData(), w_dev.getDevData(),
         m, n, d,
         1,
         mw,
         da_dev.getDevData(),
         1 );

   //---------------
   // stop gpu timer
   //---------------
   float elapsedTime_dev = stopDevTimer();
   cout << "GPU timer time: " << elapsedTime_dev << "ms" << endl;

   Matrix da2 = rand_col_matrix( d, n );
   da_dev.copyToHost( da2 );
#ifdef PRINT_MATRIX
   print_matrix( da2, "da after dev compute" );
#endif
   // diff
   compare_matrix( da, da2, "da-da2" );
   // print speed up
   cout << "speed up: " << elapsedTime_host/elapsedTime_dev << endl;

   //-----------------------------------------
   //    dev code: lower bound 
   //    cublas + copy time/2
   //-----------------------------------------
   // compute wx+b
   startDevTimer();
   NVMatrix& wt_dev = w_dev.getTranspose();
   da_dev.addProduct( v_dev, wt_dev, 0, 1 );
   delete &wt_dev;
   float elapsedTime_cublas = stopDevTimer();

   // copy data: read/write
   MaskWeights mw2( 0.5 );
   mw2.resize( m, n, d );
   startDevTimer();
   checkCuda( cudaMemcpy2D( 
       const_cast<unsigned int*>( mw2.data() ), 
       mw2.stride() * sizeof( unsigned int ),
       mw.data(),
       mw.stride() * sizeof( unsigned int ),
       mw.get_width() * sizeof( unsigned int ),
       mw.get_height(), cudaMemcpyDeviceToDevice ) );
   // divide 2 because copy includes read and write
   elapsedTime_cublas += stopDevTimer()/2; 
   cout << "Lower Bound(cublas+read memory): " << 
       elapsedTime_cublas << "ms" << endl;
}

void test_bpropw() {
   //int m = 5;
   //int n = 8;
   //int d = 3;

   int m = 1024;
   int n = 1024;
   int d = 128;

   //int m = 900;
   //int n = 807;
   //int d = 110;

   //-----------------------------------------
   // set up maskWeight
   //-----------------------------------------
   MaskWeights mw;
   mw.resize( m, n, d );
   mw.randomize();
   vector<unsigned int> mw_host;
   copyDevMaskWeights( mw, mw_host );
#ifdef PRINT_MATRIX
   print_mw( mw_host, mw.get_width(), mw.get_height() );
#endif

   //-----------------------------------------
   //        host code
   //-----------------------------------------
   // input
   Matrix v = rand_col_matrix( d, m ) ;
   Matrix a = rand_col_matrix( d, n );

   // output
   Matrix dw = rand_col_matrix( n, m );
   Matrix dw_prev( dw );
   dw_prev.setTrans(true);
   dw.copy( dw_prev );

   //---------------
   // start gpu timer
   //---------------
   startDevTimer();

   // call bpropw
   computeFCDropC_bit_bpropWeights_h(
         a.getData(), v.getData(),
         m, n, d,
         1,
         &mw_host[0],
         dw.getData(), 1
         );

   //---------------
   // stop gpu timer
   //---------------
   float elapsedTime_host = stopDevTimer();
   cout << "GPU timer time: " << elapsedTime_host << "ms" << endl;

   // print output
#ifdef PRINT_MATRIX
   print_matrix( dw, "dw after host compute" );
#endif

   //-----------------------------------------
   //        dev code
   //-----------------------------------------
   // input
   NVMatrix v_dev( d, m, true );
   v_dev.copyFromHost( v );

   NVMatrix a_dev( d, n, true );
   a_dev.copyFromHost( a );

   // output
   NVMatrix dw_dev( n, m, true );
   dw_dev.copyFromHost( dw_prev );

   //---------------
   // start gpu timer
   //---------------
   startDevTimer();

   //---------------
   // call bpropw
   //---------------
   computeFCDropC_bit_bpropWeights_d(
         a_dev.getDevData(), v_dev.getDevData(),
         m, n, d,
         1,
         mw,
         dw_dev.getDevData(), 1
         );
   //---------------
   // stop gpu timer
   //---------------
   float elapsedTime_dev = stopDevTimer();
   cout << "GPU timer time: " << elapsedTime_dev << "ms" << endl;

   Matrix dw2 = rand_col_matrix(  n, m );
   dw_dev.copyToHost( dw2 );

#ifdef PRINT_MATRIX
   print_matrix( dw2, "dw after dev compute" );
#endif
   // diff
   compare_matrix( dw, dw2, "dw-dw2" );
   // print speed up
   cout << "speed up: " << elapsedTime_host/elapsedTime_dev << endl;

   //-----------------------------------------
   //    dev code: lower bound 
   //    cublas + copy time/2
   //-----------------------------------------
   // compute wx+b
   startDevTimer();
   NVMatrix& at_dev = a_dev.getTranspose();
   dw_dev.addProduct( at_dev, v_dev, 0, 1 );
   delete &at_dev;
   float elapsedTime_cublas = stopDevTimer();

   // copy data: read/write
   MaskWeights mw2( 0.5 );
   mw2.resize( m, n, d );
   startDevTimer();
   checkCuda( cudaMemcpy2D( 
       const_cast<unsigned int*>( mw2.data() ), 
       mw2.stride() * sizeof( unsigned int ),
       mw.data(),
       mw.stride() * sizeof( unsigned int ),
       mw.get_width() * sizeof( unsigned int ),
       mw.get_height(), cudaMemcpyDeviceToDevice ) );
   // divide 2 because copy includes read and write
   elapsedTime_cublas += stopDevTimer()/2; 
   cout << "Lower Bound(cublas+read memory): " << 
       elapsedTime_cublas << "ms" << endl;
}

int main( int argc, char* argv[] ) {
    if( argc > 1 ) {
        int devId = atoi( argv[1] );
        checkCuda( cudaSetDevice( devId ) );
        cout << "Manually Set Device: " << devId << endl;
    }
   srand(0);
   //test_inference();
   test_maskweights();
   test_fprop();
   test_bpropa();
   test_bpropw();
   return 0;
}
