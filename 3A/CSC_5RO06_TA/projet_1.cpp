#include <iostream>
#include <Eigen/Dense>
#include <cstdlib>
#include <ctime>
#include <chrono>

using namespace std;
using namespace Eigen;

typedef Matrix<int, Dynamic, Dynamic> IntMatrix;

// Algorithme 1 : Multiplication naive
IntMatrix naiveMultiplication(const IntMatrix& A, const IntMatrix& B) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Initialize matrix C with zeros
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}


// Algorithme 2 : Multiplication naive avec boucle réordonnées
IntMatrix naiveReorderedMultiplication(const IntMatrix& A, const IntMatrix& B) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Initialize matrix C with zeros
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k)
            for (int j = 0; j < n; ++j)
                C(i, j) += A(i, k) * B(k, j);
    return C;
}

// Algorithme 3 : Multiplication par blocs
IntMatrix blockMultiplication(const IntMatrix& A, const IntMatrix& B, int blockSize) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Initialize matrix C with zeros
    for (int ii = 0; ii < n; ii += blockSize)
        for (int jj = 0; jj < n; jj += blockSize)
            for (int kk = 0; kk < n; kk += blockSize)
                for (int i = ii; i < min(ii + blockSize, n); ++i)
                    for (int j = jj; j < min(jj + blockSize, n); ++j)
                        for (int k = kk; k < min(kk + blockSize, n); ++k)
                            C(i, j) += A(i, k) * B(k, j);
    return C;
}


// Algorithme 4 : Multiplication par blocs reordonnée
IntMatrix blockReorderedMultiplication(const IntMatrix& A, const IntMatrix& B, int blockSize) {
    int n = A.rows();
    IntMatrix C(n, n);
    C.setZero(); // Initialize matrix C with zeros
    for (int ii = 0; ii < n; ii += blockSize)
        for (int kk = 0; kk < n; kk += blockSize)
            for (int jj = 0; jj < n; jj += blockSize)
                for (int i = ii; i < min(ii + blockSize, n); ++i)
                    for (int k = kk; k < min(kk + blockSize, n); ++k)
                        for (int j = jj; j < min(jj + blockSize, n); ++j)
                            C(i, j) += A(i, k) * B(k, j);
    return C;
}

// Algorithme 5 : Multiplication avec la bibliothèque Eigen
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

// Función para restar matrices de tipo IntMatrix
IntMatrix subtractMatrix(const IntMatrix& A, const IntMatrix& B) {
    if (A.rows() != B.rows() || A.cols() != B.cols()) {
        throw std::invalid_argument("Matrix sizes do not match.");
    }
    return A - B;
}



// Strassen para combinar matrices
void multiplyMatrixStrassen(const IntMatrix& A, const IntMatrix& B, IntMatrix& C, int n) {
    if (n == 1) {
        C(0, 0) = A(0, 0) * B(0, 0);
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
    multiplyMatrixStrassen(AResult, BResult, M1, newSize);

    // M2 = (A21 + A22) * B11
    AResult = addMatrix(A21, A22);
    multiplyMatrixStrassen(AResult, B11, M2, newSize);

    // M3 = A11 * (B12 - B22)
    BResult = subtractMatrix(B12, B22);
    multiplyMatrixStrassen(A11, BResult, M3, newSize);

    // M4 = A22 * (B21 - B11)
    BResult = subtractMatrix(B21, B11);
    multiplyMatrixStrassen(A22, BResult, M4, newSize);

    // M5 = (A11 + A12) * B22
    AResult = addMatrix(A11, A12);
    multiplyMatrixStrassen(AResult, B22, M5, newSize);

    // M6 = (A21 - A11) * (B11 + B12)
    AResult = subtractMatrix(A21, A11);
    BResult = addMatrix(B11, B12);
    multiplyMatrixStrassen(AResult, BResult, M6, newSize);

    // M7 = (A12 - A22) * (B21 + B22)
    AResult = subtractMatrix(A12, A22);
    BResult = addMatrix(B21, B22);
    multiplyMatrixStrassen(AResult, BResult, M7, newSize);

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

void printMatrix(const IntMatrix& matrix) {
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            cout << matrix(i, j) << " ";
        }
        cout << endl;
    }
}

bool compareMatrices(const IntMatrix& A, const IntMatrix& B) {
    return A == B; // Check if the matrices are approximately equal
}


int main() {
    vector<int> sizes = {8, 64, 256, 512, 1024};

    for (int n : sizes) {
        cout << "Ejecutando para n = " << n << endl;

        srand(time(0)); // Initialize the seed for random numbers

        // Generate random numbers between 1 and 10
        IntMatrix A = IntMatrix::NullaryExpr(n, n, []() { return rand() % 10 + 1; });
        IntMatrix B = IntMatrix::NullaryExpr(n, n, []() { return rand() % 10 + 1; });
        IntMatrix C5(n, n);

        // Naive multiplication
        auto start = chrono::high_resolution_clock::now();
        IntMatrix C1 = naiveMultiplication(A, B);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion naive: " << elapsed.count() << " segundos" << endl;

        // Naive reordered multiplication
        start = chrono::high_resolution_clock::now();
        IntMatrix C2 = naiveReorderedMultiplication(A, B);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion naive reordenada: " << elapsed.count() << " segundos" << endl;

        // Block multiplication
        start = chrono::high_resolution_clock::now();
        IntMatrix C3 = blockMultiplication(A, B, 2);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion por bloques: " << elapsed.count() << " segundos" << endl;

        // Block reordered multiplication
        start = chrono::high_resolution_clock::now();
        IntMatrix C3_2 = blockReorderedMultiplication(A, B, 2);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion por bloques reordenada: " << elapsed.count() << " segundos" << endl;

        // Eigen multiplication
        MatrixXd A_double = A.cast<double>();
        MatrixXd B_double = B.cast<double>();
        start = chrono::high_resolution_clock::now();
        MatrixXd C4_double = eigenMultiplication(A_double, B_double);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion con Eigen: " << elapsed.count() << " segundos" << endl;
        IntMatrix C4 = C4_double.cast<int>();

        // Strassen multiplication
        start = chrono::high_resolution_clock::now();
        multiplyMatrixStrassen(A, B, C5, n);
        end = chrono::high_resolution_clock::now();
        elapsed = end - start;
        cout << "Tiempo de ejecucion de la multiplicacion con Strassen: " << elapsed.count() << " segundos" << endl;
        /*
        // Comparar resultados con C1
        cout << "Comparacion con C1:" << endl;
        cout << "C2: " << (compareMatrices(C1, C2) ? "yes" : "no") << endl;
        cout << "C3: " << (compareMatrices(C1, C3) ? "yes" : "no") << endl;
        cout << "C3_2: " << (compareMatrices(C1, C3_2) ? "yes" : "no") << endl;
        cout << "C4: " << (compareMatrices(C1, C4) ? "yes" : "no") << endl;
        cout << "C5: " << (compareMatrices(C1, C5) ? "yes" : "no") << endl;
        */
        cout << endl;
    }

    return 0;
}
