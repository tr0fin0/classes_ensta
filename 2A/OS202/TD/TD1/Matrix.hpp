#ifndef _MATRIX_HPP_
# define _MATRIX_HPP_

# include <vector>

class Matrix
{
public:
  // Constructors - destructor
  Matrix(int nRows, int nCols);
  Matrix(int nRows, int nCols, double val);
  Matrix(const Matrix & A) = delete;
  Matrix(Matrix && A) = default;
  ~Matrix() = default;

  // Operators
  Matrix & operator =(const Matrix & A) = delete;
  Matrix & operator =(Matrix && A) = default;

  // Getters - Setters 
  double operator() (int i, int j) const
  {
    return m_arr_coefs[i+j*nbRows];
  }

  double &operator() (int i, int j)
  {
    return m_arr_coefs[i+j*nbRows];
  }

  double const* data() const { return m_arr_coefs.data(); }
  double      * data()       { return m_arr_coefs.data(); }
  
  int nbRows, nbCols;
private:
  std::vector < double >m_arr_coefs;
};

#endif