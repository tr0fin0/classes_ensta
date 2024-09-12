#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iomanip>  // std::setprecision, std::setw
#include <vector>
#include "Matrix.hpp"
#include "ProdMatMat.hpp"


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

void printLine(float dim, float time, float instr)
{
  std::cout << std::fixed
            << std::setprecision(0)
            << std::setw(4) << dim  << " & "
            << std::fixed
            << std::setprecision(6)
            << std::setw(5) << time << " & "
            << std::fixed
            << std::setprecision(2)
            << std::setw(5) << instr << "\\\\" << std::endl;

}


int main(int nargs, char *vargs[])
{
  int dim = 1024;
  if (nargs > 1)
    dim = atoi(vargs[1]); // dim gets value from X when ./TestProductMatrix.exe X

  // std::vector <int> dims{1023, 1024, 1025, 2047, 2048, 2049};
  std::vector <int> dims{1024, 2048};
  float times  = 0;
  float instrs = 0;

  bool isPassed = true;
  for(int i = 0; i < dims.size(); i++)
  {
    dim = dims[i];

    std::vector < double >uA, vA, uB, vB;
    std::tie(uA, vA, uB, vB) = computeTensors(dim);

    Matrix A = initTensorMatrices(uA, vA);
    Matrix B = initTensorMatrices(uB, vB);

    std::chrono::time_point < std::chrono::system_clock > start, end;
    start = std::chrono::system_clock::now();
    Matrix C = A * B;
    end = std::chrono::system_clock::now();
    std::chrono::duration < double >elapsed_seconds = end - start;


    isPassed = verifProduct(uA, vA, uB, vB, C);
    if (isPassed)
      {
        // std::cout << "Test passed\n";
        // std::cout << "Temps CPU produit matrice-matrice naif : " << elapsed_seconds.count() << " secondes\n";
        // std::cout << "MFlops -> " << (2.*dim*dim*dim)/elapsed_seconds.count()/1000000 <<std::endl;
        // std::cout << "[dim: " << dim << ", time: " << elapsed_seconds.count() << " s, MFloops: " << (2.*dim*dim*dim)/elapsed_seconds.count()/1000000 << "]" << std::endl;

        float MFloops = (2.*dim*dim*dim)/elapsed_seconds.count()/1000000;
        float time = elapsed_seconds.count();
        printLine(dim, time, MFloops);

        times  += time;
        instrs += MFloops;
      }
    else
      std::cout << "Test failed\n";
  }
  std::cout << "\\hline\n" 
            << "avg  & "
            << std::fixed
            << std::setprecision(6)
            << std::setw(5) << (times/dims.size()) << " & "
            << std::fixed
            << std::setprecision(2)
            << std::setw(5) << (instrs/dims.size()) << "\\\\" << std::endl;

  return (isPassed ? EXIT_SUCCESS : EXIT_FAILURE);
}