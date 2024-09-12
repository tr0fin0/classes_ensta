#include <algorithm>
#include <cassert>
#include <iostream>
#include <thread>
#if defined(_OPENMP)
#include <omp.h>
#endif
#include "ProdMatMat.hpp"


namespace {
  void prodSubBlocks(int iRowBlkA, int iColBlkB, int iColBlkA, int szBlock, const Matrix& A, const Matrix& B, Matrix& C) {

    #pragma omp parallel        // parallel declaration
    {
      int i, j, k;
      omp_set_num_threads(16);  // parallel threads
                                // 16 threads, most efficient
      #pragma omp for           // declare for function as parallel
                                // j k i order, most efficient 
      for (j = iColBlkB; j < std::min(B.nbCols, iColBlkB + szBlock); j++)
        for (k = iColBlkA; k < std::min(A.nbCols, iColBlkA + szBlock); k++)
          for (i = iRowBlkA; i < std::min(A.nbRows, iRowBlkA + szBlock); ++i)
            C(i, j) += A(i, k) * B(k, j);
    // const int szBlock = 32;
    }
  }
}  // namespace


namespace {
  void prodBlocks(int sizeBlock, const Matrix& A, const Matrix& B, Matrix& C) {

    #pragma omp parallel        // parallel declaration
    {
      int dim = std::max({A.nbRows, B.nbCols, A.nbCols});

      int m, n, i, j, k;
      // where:
      // m: row of blocks
      // n: columns of blocks
      // i: row inside of block
      // j: column inside of block
      // k: accumulator inside of multiplication
      omp_set_num_threads(16);  // parallel declaration

      #pragma omp for           // parallel declaration
      for (n = 0; n < dim; n+=sizeBlock)
        for (m = 0; m < dim; m+=sizeBlock)
          for (j = n; j < n+sizeBlock; j++)
            for (k = 0; k < dim; k++)
              for (i = m; i < m+sizeBlock; i++)
                C(i, j) += A(i, k) * B(k, j);
    // const int szBlock = 32;
    }
  }
}  // namespace

Matrix operator*(const Matrix& A, const Matrix& B) {
  Matrix C(A.nbRows, B.nbCols, 0.0);
  // std::cout << "dim: " << std::max({A.nbRows, B.nbCols, A.nbCols}) << std::endl;
  prodBlocks(512, A, B, C);  // with block
  // prodSubBlocks(0, 0, 0, std::max({A.nbRows, B.nbCols, A.nbCols}), A, B, C);  // without blocking
  return C;
}