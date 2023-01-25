# include "Matrix.hpp"
# include <cassert>

Matrix::Matrix( int nRows, int nCols ) :
  nbRows{nRows}, nbCols{nCols}, m_arr_coefs(nRows*nCols)
{}
// ------------------------------------------------------------------------
Matrix::Matrix( int nRows, int nCols, double val ) :
  nbRows{nRows}, nbCols{nCols}, m_arr_coefs(nRows*nCols, val)
{}
// ========================================================================