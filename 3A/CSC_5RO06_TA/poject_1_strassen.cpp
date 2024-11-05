#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
using namespace Eigen;

typedef Matrix<int, Dynamic, Dynamic> IntMatrix;

// Función para sumar matrices de tipo IntMatrix
IntMatrix addMatrix(const IntMatrix& A, const IntMatrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix sizes do not match.");
    }
    return A + B;
}

// Función para restar matrices de tipo IntMatrix
IntMatrix subtractMatrix(const IntMatrix& A, const IntMatrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix sizes do not match.");
    }
    return A - B;
}

// Función para multiplicación naive
void multiplyMatrixNaive(const IntMatrix& A, const IntMatrix& B, IntMatrix& C, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            C(i, j) = 0;
            for (int k = 0; k < n; ++k) {
                C(i, j) += A(i, k) * B(k, j);
            }
        }
    }
}

// Strassen para combinar matrices con límite de profundidad
void multiplyMatrixStrassen(const IntMatrix& A, const IntMatrix& B, IntMatrix& C, int n, int depth = 0, int maxDepth = 2) {
    if (n == 1) {
        C(0, 0) = A(0, 0) * B(0, 0);
        return;
    }

    if (depth >= maxDepth) {
        multiplyMatrixNaive(A, B, C, n);
        return;
    }

    int newSize = n / 2;
    IntMatrix A11 = A.topLeftCorner(newSize, newSize);
    IntMatrix A12 = A.topRightCorner(newSize, newSize);
    IntMatrix A21 = A.bottomLeftCorner(newSize, newSize);
    IntMatrix A22 = A.bottomRightCorner(newSize, newSize);

    IntMatrix B11 = B.topLeftCorner(newSize, newSize);
    IntMatrix B12 = B.topRightCorner(newSize, newSize);
    IntMatrix B21 = B.bottomLeftCorner(newSize, newSize);
    IntMatrix B22 = B.bottomRightCorner(newSize, newSize);

    IntMatrix C11(newSize, newSize);
    IntMatrix C12(newSize, newSize);
    IntMatrix C21(newSize, newSize);
    IntMatrix C22(newSize, newSize);

    IntMatrix M1(newSize, newSize);
    IntMatrix M2(newSize, newSize);
    IntMatrix M3(newSize, newSize);
    IntMatrix M4(newSize, newSize);
    IntMatrix M5(newSize, newSize);
    IntMatrix M6(newSize, newSize);
    IntMatrix M7(newSize, newSize);

    IntMatrix AResult(newSize, newSize);
    IntMatrix BResult(newSize, newSize);

    // M1 = (A11 + A22) * (B11 + B22)
    AResult = addMatrix(A11, A22);
    BResult = addMatrix(B11, B22);
    multiplyMatrixStrassen(AResult, BResult, M1, newSize, depth + 1, maxDepth);

    // M2 = (A21 + A22) * B11
    AResult = addMatrix(A21, A22);
    multiplyMatrixStrassen(AResult, B11, M2, newSize, depth + 1, maxDepth);

    // M3 = A11 * (B12 - B22)
    BResult = subtractMatrix(B12, B22);
    multiplyMatrixStrassen(A11, BResult, M3, newSize, depth + 1, maxDepth);

    // M4 = A22 * (B21 - B11)
    BResult = subtractMatrix(B21, B11);
    multiplyMatrixStrassen(A22, BResult, M4, newSize, depth + 1, maxDepth);

    // M5 = (A11 + A12) * B22
    AResult = addMatrix(A11, A12);
    multiplyMatrixStrassen(AResult, B22, M5, newSize, depth + 1, maxDepth);

    // M6 = (A21 - A11) * (B11 + B12)
    AResult = subtractMatrix(A21, A11);
    BResult = addMatrix(B11, B12);
    multiplyMatrixStrassen(AResult, BResult, M6, newSize, depth + 1, maxDepth);

    // M7 = (A12 - A22) * (B21 + B22)
    AResult = subtractMatrix(A12, A22);
    BResult = addMatrix(B21, B22);
    multiplyMatrixStrassen(AResult, BResult, M7, newSize, depth + 1, maxDepth);

    // C11 = M1 + M4 - M5 + M7
    C11 = addMatrix(addMatrix(M1, M4), subtractMatrix(M7, M5));

    // C12 = M3 + M5
    C12 = addMatrix(M3, M5);

    // C21 = M2 + M4
    C21 = addMatrix(M2, M4);

    // C22 = M1 - M2 + M3 + M6
    C22 = addMatrix(subtractMatrix(M1, M2), addMatrix(M3, M6));

    // Combinar submatrices en C
    C.topLeftCorner(newSize, newSize) = C11;
    C.topRightCorner(newSize, newSize) = C12;
    C.bottomLeftCorner(newSize, newSize) = C21;
    C.bottomRightCorner(newSize, newSize) = C22;
}

int main() {
    vector<int> sizes = {8, 64, 256, 512, 1024};

    for (int n : sizes) {
        cout << "Ejecutando para n = " << n << endl;

        srand(time(0)); // Initialize the seed for random numbers

        // Generate random numbers between 1 and 10
        IntMatrix A = IntMatrix::NullaryExpr(n, n, []() { return rand() % 10 + 1; });
        IntMatrix B = IntMatrix::NullaryExpr(n, n, []() { return rand() % 10 + 1; });
        IntMatrix C(n, n);

        for (int depth = 1; depth <= 4; ++depth) {
            cout << "Profundidad de recursion: " << depth << endl;
            auto start = chrono::high_resolution_clock::now();
            multiplyMatrixStrassen(A, B, C, n, 0, depth);
            auto end = chrono::high_resolution_clock::now();
            chrono::duration<double> elapsed = end - start;
            cout << "Tiempo de ejecucion de la multiplicacion con Strassen (profundidad " << depth << "): " << elapsed.count() << " segundos" << endl;
        }
        cout << endl;
    }

    return 0;
}