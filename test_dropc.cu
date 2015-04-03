/**
 *  dropc kerenel testing code
 */

#include <cassert>
#include <iostream>
#include <string>
#include "libs/matrix.h"
#include "libs/nvmatrix.cuh"
#include "libs/cuda_common.hpp"

#include "dropc_host.hpp"
#include "dropc_dev.hpp"

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
//---------------------------------------------------------------------


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

    // masks
    Matrix mw = rand_col_matrix( n, m*d);
    mw.biggerThanScalar( 0.5 );
    //print_matrix( mw, "wb matrix" );

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
    computeFCDropC_fprop_h( 
            x.getData(), w.getData(), b.getData(), // input matrix
            m, n, d, // dims
            mw.getData(), mb.getData(),  // masks
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
    NVMatrix mw_dev( n, m*d, true );
    mw_dev.copyFromHost( mw );

    NVMatrix mb_dev( d, m, true );
    mb_dev.copyFromHost( mb );

    // output
    NVMatrix y_dev( d, m, true );
    y_dev.copyFromHost( y_prev );

    //print_matrix( y_dev, "y_dev before dev compute" );
    //---------------
    // start gpu timer
    //---------------
    startDevTimer();

    //--------------
    // call fprop
    //--------------
    computeFCDropC_fprop_d(
            x_dev.getDevData(), w_dev.getDevData(), b_dev.getDevData(), // input matrix
            m, n, d, // dims
            mw_dev.getDevData(), mb_dev.getDevData(),  // masks
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
    //        host code
    //-----------------------------------------
    // input
    Matrix v = rand_col_matrix( d,m );
    Matrix w = rand_col_matrix( n,m );

    // masks
    Matrix mw = rand_col_matrix( n, m*d);
    mw.biggerThanScalar( 0.5 );

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
    computeFCDropC_bpropActs_h(
            v.getData(), w.getData(),
            m, n, d,
            1,
            mw.getData(),
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

    // masks
    NVMatrix mw_dev( n, m*d, true );
    mw_dev.copyFromHost( mw );

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
    computeFCDropC_bpropActs_d(
            v_dev.getDevData(), w_dev.getDevData(),
            m, n, d,
            1,
            mw_dev.getDevData(),
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
    //        host code
    //-----------------------------------------
    // input
    Matrix v = rand_col_matrix( d, m ) ;
    Matrix a = rand_col_matrix( d, n );

    // masks
    Matrix mw = rand_col_matrix( n, m*d);
    mw.biggerThanScalar( 0.5 );

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
    computeFCDropC_bpropWeights_h(
            a.getData(), v.getData(),
            m, n, d,
            1,
            mw.getData(),
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

    // masks
    NVMatrix mw_dev( n, m*d, true );
    mw_dev.copyFromHost( mw );

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
    computeFCDropC_bpropWeights_d(
            a_dev.getDevData(), v_dev.getDevData(),
            m, n, d,
            1,
            mw_dev.getDevData(),
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
}

int main(int argc, char* argv[] ) {
    if( argc > 1 ) {
        int devId = atoi( argv[1] );
        checkCuda( cudaSetDevice( devId ) );
        cout << "Manually Set Device: " << devId << endl;
    }
    test_fprop();
    test_bpropa();
    test_bpropw();
    return 0;
}
