#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h> // Incluir OpenMP

using namespace std;
using namespace Eigen;

typedef Matrix<int, Dynamic, Dynamic> IntMatrix;

// Algoritmo 1 : Multiplicación naive
IntMatrix naiveMultiplication(const IntMatrix& A, const IntMatrix& B) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Inicializar la matriz C con ceros
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}

// Algoritmo 2 : Multiplicación naive con bucles reordenados
IntMatrix naiveReorderedMultiplication(const IntMatrix& A, const IntMatrix& B) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Inicializar la matriz C con ceros
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}

// Algoritmo 3 : Multiplicación por bloques
IntMatrix blockMultiplication(const IntMatrix& A, const IntMatrix& B, int blockSize) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Inicializar la matriz C con ceros
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += blockSize)
        for (int jj = 0; jj < n; jj += blockSize)
            for (int kk = 0; kk < n; kk += blockSize)
                for (int i = ii; i < min(ii + blockSize, n); ++i)
                    for (int j = jj; j < min(jj + blockSize, n); ++j)
                        for (int k = kk; k < min(kk + blockSize, n); ++k)
                            C(i, j) += A(i, k) * B(k, j);
    return C;
}

// Algoritmo 4 : Multiplicación por bloques reordenada
IntMatrix blockReorderedMultiplication(const IntMatrix& A, const IntMatrix& B, int blockSize) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Inicializar la matriz C con ceros
    #pragma omp parallel for collapse(2)
    for (int ii = 0; ii < n; ii += blockSize)
        for (int kk = 0; kk < n; kk += blockSize)
            for (int jj = 0; jj < n; jj += blockSize)
                for (int i = ii; i < min(ii + blockSize, n); ++i)
                    for (int k = kk; k < min(kk + blockSize, n); ++k)
                        for (int j = jj; j < min(jj + blockSize, n); ++j)
                            C(i, j) += A(i, k) * B(k, j);
    return C;
}

// Algoritmo 5 : Multiplicación con la biblioteca Eigen
MatrixXd eigenMultiplication(const MatrixXd& A, const MatrixXd& B) {
    return A * B;
}

// Función para sumar matrices de tipo IntMatrix
IntMatrix addMatrix(const IntMatrix& A, const IntMatrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix sizes do not match.");
    }
    return A + B;
}


void printMatrix(const IntMatrix& matrix) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            cout << matrix(i, j) << " ";
        }
        cout << endl;
    }
}

bool compareMatrices(const IntMatrix& A, const IntMatrix& B) {
    return A == B; // Verificar si las matrices son aproximadamente iguales
}

int main() {
    vector<int> sizes = {8, 64, 256, 512, 1024};

    omp_set_num_threads(8);

    for (int n : sizes) {
        cout << "Ejecutando para n = " << n << endl;

        srand(time(0)); // Inicializar la semilla para números aleatorios

        // Generar números aleatorios entre 1 y 10
        IntMatrix A = IntMatrix::NullaryExpr(n, n, []() { return rand() % 10 + 1; });
        IntMatrix B = IntMatrix::NullaryExpr(n, n, []() { return rand() % 10 + 1; });

        // Multiplicación naive
        auto start = chrono::high_resolution_clock::now();
        IntMatrix C1 = naiveMultiplication(A, B);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion naive: " << elapsed.count() << " segundos" << endl;

        // Multiplicación naive reordenada
        start = chrono::high_resolution_clock::now();
        IntMatrix C2 = naiveReorderedMultiplication(A, B);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion naive reordenada: " << elapsed.count() << " segundos" << endl;

        // Multiplicación por bloques
        start = chrono::high_resolution_clock::now();
        IntMatrix C3 = blockMultiplication(A, B, 2);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion por bloques: " << elapsed.count() << " segundos" << endl;

        // Multiplicación por bloques reordenada
        start = chrono::high_resolution_clock::now();
        IntMatrix C3_2 = blockReorderedMultiplication(A, B, 2);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion por bloques reordenada: " << elapsed.count() << " segundos" << endl;

        cout << endl;
    }

    return 0;
}