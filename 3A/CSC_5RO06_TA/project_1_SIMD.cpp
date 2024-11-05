#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <immintrin.h> // Incluir cabecera para AVX

using namespace std;
using namespace Eigen;

typedef Matrix<int, Dynamic, Dynamic> IntMatrix;

// Algoritmo 1: Multiplicación naive con SIMD
IntMatrix naiveMultiplicationSIMD(const IntMatrix& A, const IntMatrix& B) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Inicializar matriz C con ceros

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            __m256i sum = _mm256_setzero_si256(); // Inicializar sumador SIMD
            for (int k = 0; k < n; k += 8) {
                __m256i a = _mm256_loadu_si256((__m256i*)&A(i, k)); // Cargar 8 elementos de A
                __m256i b = _mm256_loadu_si256((__m256i*)&B(k, j)); // Cargar 8 elementos de B
                __m256i prod = _mm256_mullo_epi32(a, b); // Multiplicar elementos
                sum = _mm256_add_epi32(sum, prod); // Sumar productos
            }
            // Sumar elementos del vector SIMD
            int temp[8];
            _mm256_storeu_si256((__m256i*)temp, sum);
            for (int l = 0; l < 8; ++l) {
                C(i, j) += temp[l];
            }
        }
    }
    return C;
}

// Algorithme 2 : Multiplication naive avec boucle réordonnées avec SIMD
IntMatrix naiveReorderedMultiplicationSIMD(const IntMatrix& A, const IntMatrix& B) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Initialize matrix C with zeros
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            __m256i a = _mm256_set1_epi32(A(i, k)); // Broadcast A(i, k) to all elements of a
            for (int j = 0; j < n; j += 8) {
                __m256i b = _mm256_loadu_si256((__m256i*)&B(k, j)); // Load 8 elements from B
                __m256i c = _mm256_loadu_si256((__m256i*)&C(i, j)); // Load 8 elements from C
                __m256i prod = _mm256_mullo_epi32(a, b); // Multiply elements
                c = _mm256_add_epi32(c, prod); // Add products to C
                _mm256_storeu_si256((__m256i*)&C(i, j), c); // Store result back to C
            }
        }
    }
    return C;
}

IntMatrix blockMultiplicationSIMD(const IntMatrix& A, const IntMatrix& B, int blockSize) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Initialize matrix C with zeros
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int jj = 0; jj < n; jj += blockSize) {
            for (int kk = 0; kk < n; kk += blockSize) {
                for (int i = ii; i < min(ii + blockSize, n); ++i) {
                    for (int j = jj; j < min(jj + blockSize, n); ++j) {
                        __m256i sum = _mm256_setzero_si256(); // Initialize SIMD sum
                        for (int k = kk; k < min(kk + blockSize, n); k += 8) {
                            __m256i a = _mm256_loadu_si256((__m256i*)&A(i, k)); // Load 8 elements from A
                            __m256i b = _mm256_loadu_si256((__m256i*)&B(k, j)); // Load 8 elements from B
                            __m256i prod = _mm256_mullo_epi32(a, b); // Multiply elements
                            sum = _mm256_add_epi32(sum, prod); // Add products
                        }
                        // Sum elements of the SIMD vector
                        int temp[8];
                        _mm256_storeu_si256((__m256i*)temp, sum);
                        for (int l = 0; l < 8; ++l) {
                            C(i, j) += temp[l];
                        }
                    }
                }
            }
        }
    }
    return C;
}


// Algorithme 4 : Multiplication par blocs reordonnée avec SIMD
IntMatrix blockReorderedMultiplicationSIMD(const IntMatrix& A, const IntMatrix& B, int blockSize) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Initialize matrix C with zeros
    for (int ii = 0; ii < n; ii += blockSize) {
        for (int kk = 0; kk < n; kk += blockSize) {
            for (int jj = 0; jj < n; jj += blockSize) {
                for (int i = ii; i < min(ii + blockSize, n); ++i) {
                    for (int k = kk; k < min(kk + blockSize, n); ++k) {
                        __m256i a = _mm256_set1_epi32(A(i, k)); // Broadcast A(i, k) to all elements of a
                        for (int j = jj; j < min(jj + blockSize, n); j += 8) {
                            __m256i b = _mm256_loadu_si256((__m256i*)&B(k, j)); // Load 8 elements from B
                            __m256i c = _mm256_loadu_si256((__m256i*)&C(i, j)); // Load 8 elements from C
                            __m256i prod = _mm256_mullo_epi32(a, b); // Multiply elements
                            c = _mm256_add_epi32(c, prod); // Add products to C
                            _mm256_storeu_si256((__m256i*)&C(i, j), c); // Store result back to C
                        }
                    }
                }
            }
        }
    }
    return C;
}

int main() {
    vector<int> sizes = {8, 64, 256, 512, 1024};

    for (int n : sizes) {
        cout << "Ejecutando para n = " << n << endl;

        srand(time(0)); // Initialize the seed for random numbers

        // Generate random numbers between 1 and 10
        IntMatrix A = IntMatrix::NullaryExpr(n, n, []() { return rand() % 10 + 1; });
        IntMatrix B = IntMatrix::NullaryExpr(n, n, []() { return rand() % 10 + 1; });

        // Naive multiplication
        auto start = chrono::high_resolution_clock::now();
        IntMatrix C1 = naiveMultiplicationSIMD(A, B);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion naive: " << elapsed.count() << " segundos" << endl;

        // Naive reordered multiplication
        start = chrono::high_resolution_clock::now();
        IntMatrix C2 = naiveReorderedMultiplicationSIMD(A, B);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion naive reordenada: " << elapsed.count() << " segundos" << endl;

        // Block multiplication
        start = chrono::high_resolution_clock::now();
        IntMatrix C3 = blockMultiplicationSIMD(A, B, 2);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion por bloques: " << elapsed.count() << " segundos" << endl;

        // Block reordered multiplication
        start = chrono::high_resolution_clock::now();
        IntMatrix C3_2 = blockReorderedMultiplicationSIMD(A, B, 2);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion por bloques reordenada: " << elapsed.count() << " segundos" << endl;

        
        cout << endl;
    }

    return 0;
}