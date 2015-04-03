/**
 * debug utils header file
 */

#ifndef __DEBUG_UTILS_HPP__
#define __DEBUG_UTILS_HPP__

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>
#include <algorithm>
#include <cmath>
#include <vector>

const std::string TEST_RESULT_FOLDER = "test_result";
const float DIFF_THRESHOLD = 1e-4;

/** print host memory helper */
template<class T>
void printHostMemory_helper(
      const T* m,                      ///< [in] input matrix
      size_t width,                    ///< [in] width of matrix
      size_t height,                   ///< [in] height of matrix
      size_t depth,                    ///< [in] depth of matrix
      std::ostream& ofs ){
   using namespace std;
   T maxv = m[0];
   T minv = m[0];
   T minabsv = abs(m[0]);

   ofs << width << " " << height << " " << depth << endl;
   for( size_t y = 0; y < height * depth; y++ ) {
      for( size_t x = 0; x < width; x++ ) {
         T v = m[ y * width + x ];
         ofs << setw(8) << v << " ";
         if( v > maxv )
            maxv = v;
         if( v < minv )
            minv = v;
         if( abs(v) < minabsv )
            minabsv = abs(v);

      }
      ofs << endl;
   }
   ofs << minv << " " << maxv << " " << minabsv << endl;
}

/** print memory to file */
template<class T>
void printHostMemory( 
      const T* m,                      ///< [in] input matrix
      size_t width,                    ///< [in] width of matrix
      size_t height,                   ///< [in] height of matrix
      size_t depth,                    ///< [in] depth of matrix
      const std::string& fileName = "" ///< [in] file name to print
      ){
   // print to file
   std::ofstream ofs( ( TEST_RESULT_FOLDER + "/" + fileName).c_str() );
   if( ofs.good() ) {
      printHostMemory_helper( m, width, height, depth, ofs );
      ofs.close();
   }
   else { // print to cout
      printHostMemory_helper( m, width, height, depth, std::cout );
   }
}

/** print matrix info */
struct PrintMatrixInfo {
   int x;     ///< x index  
   int y;     ///< y index
   int z;     ///< z index
   float v1;  ///< vlaue 1
   float v2;  ///< vlaue 2
   PrintMatrixInfo( int x, int y, int z , float v1, float v2 )
      : x(x), y(y), z(z), v1(v1), v2(v2) {
   }
};

/** print file difference */
template<class T>
T printMemoryFileDiff( 
      const std::string& file1, ///< [in] file 1
      const std::string& file2, ///< [in] file 2
      const std::string& out,   ///< [in] output file
      int sx = -1,              ///< [in] start compare index of x, <0 means take default value 
      int ex = -1,              ///< [in] end   compare index of x, <0 means take default value
      int sy = -1,              ///< [in] start compare index of y, <0 means take default value
      int ey = -1,              ///< [in] end   compare index of y, <0 means take default value
      int sz = -1,              ///< [in] start compare index of z, <0 means take default value
      int ez = -1               ///< [in] end   compare index of z, <0 means take default value
      ){
   using namespace std;

   ifstream ifs1( ( TEST_RESULT_FOLDER + "/" + file1 ).c_str() );
   ifstream ifs2( ( TEST_RESULT_FOLDER + "/" + file2 ).c_str() );
   ofstream ofs_out( ( TEST_RESULT_FOLDER + "/" + out ).c_str() );

   // read size
   int w1, h1, d1;
   int w2, h2, d2;

   ifs1 >> w1;  
   ifs1 >> h1;  
   ifs1 >> d1;  
   ifs2 >> w2;  
   ifs2 >> h2;  
   ifs2 >> d2;  

   if( w1 != w2 || h1 != h2 || d1 != d2 ) {
      cerr << "differnt matrix size: " << endl  
           << "(" << w1 << "," << h1 << "," << d1 << ")" << endl 
           << "(" << w2 << "," << h2 << "," << d2 << ")" << endl; 
      return 0;
   }
   // adjust index to default value if < 0
   sx = sx < 0 ? 0 : sx;
   ex = ex < 0 ? w1 : ex;
   sy = sy < 0 ? 0 : sy;
   ey = ey < 0 ? h1 : ey;
   sz = sz < 0 ? 0 : sz;
   ez = ez < 0 ? d1 : ez;

   // index info
   ofs_out << w1 << " " << h1 << " " << d1 << endl;
   ofs_out << sx << " " << ex << endl;
   ofs_out << sy << " " << ey << endl;
   ofs_out << sz << " " << ez << endl;

   vector<PrintMatrixInfo> matrixInfo;  // value > threshold

   // read 2 matrix 
   T maxdiff = 0;
   for( int k = 0; k < d1; k++ ) {
      for( int j = 0; j < h1; j++ ) {
         for( int i = 0; i < w1; i++ ) {
            T v1;
            T v2;
            ifs1 >> v1;
            ifs2 >> v2;
            if( (k>= sz && k < ez) && (j>= sy && j < ey) && (i>= sx && i < ex) ) {
               T diff = abs(v1 - v2);
               if( diff > maxdiff )
                  maxdiff = diff;
               ofs_out << setw(5) << diff << " ";
               if( diff > DIFF_THRESHOLD ) {
                  matrixInfo.push_back( PrintMatrixInfo( i,j,k,v1,v2 ) );
               }
            }
         }
         ofs_out << endl;
      }
   }

   // print additional output
   ofs_out << maxdiff << endl;
   ofs_out << matrixInfo.size() << endl;
   for( size_t i = 0; i < matrixInfo.size(); i++ ) {
      PrintMatrixInfo& info = matrixInfo[i];
      ofs_out << info.x << " " << info.y << " " << info.z << " ";
      ofs_out << info.v1 << " " << info.v2 << endl;
   }

   // clean up
   ifs1.close();
   ifs2.close();
   ofs_out.close();

   return maxdiff;
}

#endif
