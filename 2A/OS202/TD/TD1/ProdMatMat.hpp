
#ifndef _ProdMatMat_hpp__
# define _ProdMatMat_hpp__
# include <functional>
#include "Matrix.hpp"

Matrix operator* ( const Matrix& A, const Matrix& B );

enum prod_algo { naive, block, parallel_naive, parallel_block1, parallel_block2 } ;
void setProdMatMat( prod_algo algo );
void setBlockSize( int size );
void setNbThreads( int n );
#endif