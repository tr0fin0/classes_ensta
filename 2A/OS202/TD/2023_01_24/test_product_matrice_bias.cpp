#include <cstdlib>
#include <vector>
#include <cassert>
#include <cmath>
#include <iostream>
#include <chrono>
#include "Matrix.hpp"
#include "ProdMatMat.hpp"

extern "C" void dgemm_(char const& trA, char const& trB, int const& m, int const& n, int const& k,
                       double const& alpha, double const* A, int const& ldA, double const* B,
                       int const& ldB, double const& beta, double* C, int const& ldC );

std::tuple<std::vector<double>,std::vector<double>,
	   std::vector<double>,std::vector<double>>  computeTensors(int dim)
{
  double pi = std::acos(-1.0);
  auto u1 = std::vector < double >(dim);
  auto u2 = std::vector < double >(dim);
  auto v1 = std::vector < double >(dim);
  auto v2 = std::vector < double >(dim);

  for (int i = 0; i < dim; i++)
    {
      u1[i] = std::cos(1.67 * i * pi / dim);
      u2[i] = std::sin(2.03 * i * pi / dim + 0.25);
      v1[i] = std::cos(1.23 * i * i * pi / (7.5 * dim));
      v2[i] = std::sin(0.675 * i / (3.1 * dim));
    }
  return std::make_tuple(u1, u2, v1, v2);
}

Matrix initTensorMatrices(const std::vector < double >&u, const std::vector < double >&v)
{
  Matrix A(u.size(), v.size());
  for (unsigned long irow = 0UL; irow < u.size(); ++irow)
    for (unsigned long jcol = 0UL; jcol < v.size(); ++jcol)
      A(irow, jcol) = u[irow] * v[jcol];
  return A;
}

double dot(const std::vector < double >&u, const std::vector < double >&v)
{
  assert(u.size() == v.size());
  double scal = 0.0;
  for (unsigned long i = 0UL; i < u.size(); ++i)
    scal += u[i] * v[i];
  return scal;
}

bool verifProduct(const std::vector < double >&uA, std::vector < double >&vA,
		  const std::vector < double >&uB, std::vector < double >&vB, const Matrix & C)
{
  double vAdotuB = dot(vA, uB);
  for (int irow = 0; irow < C.nbRows; irow++)
    for (int jcol = 0; jcol < C.nbCols; jcol++)
      {
	double rightVal = uA[irow] * vAdotuB * vB[jcol];
	if (std::fabs(rightVal - C(irow, jcol)) >
	    100*std::fabs(C(irow, jcol) * std::numeric_limits < double >::epsilon()))
	  {
	    std::
	      cerr << "Erreur numérique : valeur attendue pour C( " << irow << ", " << jcol
		   << " ) -> " << rightVal << " mais valeur trouvée : " << C(irow,jcol) << std::endl;
	    return false;
	  }
      }
  return true;
}

int main(int nargs, char *vargs[])
{
  int dim = 1024;
  if (nargs > 1)
    dim = atoi(vargs[1]);
  std::vector < double >uA, vA, uB, vB;
  std::tie(uA, vA, uB, vB) = computeTensors(dim);

  Matrix A = initTensorMatrices(uA, vA);
  Matrix B = initTensorMatrices(uB, vB);

  std::chrono::time_point < std::chrono::system_clock > start, end;
  start = std::chrono::system_clock::now();
  Matrix C(dim,dim);
  dgemm_('N', 'N', dim, dim, dim, 1., A.data(), dim, B.data(), dim, 0., C.data(), dim);
  end = std::chrono::system_clock::now();
  std::chrono::duration < double >elapsed_seconds = end - start;

  bool isPassed = verifProduct(uA, vA, uB, vB, C);
  if (isPassed)
    {
      std::cout << "Test passed\n";
      std::cout << "Temps CPU produit matrice-matrice blas : " << elapsed_seconds.count() << " secondes\n";
      std::cout << "MFlops -> " << (2.*dim*dim*dim)/elapsed_seconds.count()/1000000 <<std::endl;
    }
  else
    std::cout << "Test failed\n";

  return (isPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}