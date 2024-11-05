#include <iostream>
#include <cstdlib>
#include "xtime_l.h"
#include "xil_printf.h"
#include "xscutimer.h"

#define SIZE 1024

int A[SIZE][SIZE];
int B[SIZE][SIZE];
int C[SIZE][SIZE];

using namespace std;

XScuTimer Timer;

// Algoritmo 1: Multiplicación naive
void naiveMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            for (int k = 0; k < SIZE; ++k)
                C[i][j] += A[i][k] * B[k][j];
}

// Algoritmo 2: Multiplicación naive con bucles reordenados
void naiveReorderedMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i)
        for (int k = 0; k < SIZE; ++k)
            for (int j = 0; j < SIZE; ++j)
                C[i][j] += A[i][k] * B[k][j];
}

// Algoritmo 3: Multiplicación por bloques
void blockMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE], int blockSize) {
    for (int ii = 0; ii < SIZE; ii += blockSize)
        for (int jj = 0; jj < SIZE; jj += blockSize)
            for (int kk = 0; kk < SIZE; kk += blockSize)
                for (int i = ii; i < min(ii + blockSize, SIZE); ++i)
                    for (int j = jj; j < min(jj + blockSize, SIZE); ++j)
                        for (int k = kk; k < min(kk + blockSize, SIZE); ++k)
                            C[i][j] += A[i][k] * B[k][j];
}

// Algoritmo 4: Multiplicación por bloques reordenada
void blockReorderedMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE], int blockSize) {
    for (int ii = 0; ii < SIZE; ii += blockSize)
        for (int kk = 0; kk < SIZE; kk += blockSize)
            for (int jj = 0; jj < SIZE; jj += blockSize)
                for (int i = ii; i < min(ii + blockSize, SIZE); ++i)
                    for (int k = kk; k < min(kk + blockSize, SIZE); ++k)
                        for (int j = jj; j < min(jj + blockSize, SIZE); ++j)
                            C[i][j] += A[i][k] * B[k][j];
}

// Función para sumar matrices de tipo IntMatrix
void addMatrix(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            C[i][j] = A[i][j] + B[i][j];
}

// Función para restar matrices de tipo IntMatrix
void subtractMatrix(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            C[i][j] = A[i][j] - B[i][j];
}

// Multiplicación de matrices usando Strassen
void multiplyMatrixStrassen(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE], int size) {
    if (size == 1) {
        C[0][0] = A[0][0] * B[0][0];
        return;
    }

    int newSize = size / 2;
    int A11[SIZE][SIZE], A12[SIZE][SIZE], A21[SIZE][SIZE], A22[SIZE][SIZE];
    int B11[SIZE][SIZE], B12[SIZE][SIZE], B21[SIZE][SIZE], B22[SIZE][SIZE];
    int C11[SIZE][SIZE], C12[SIZE][SIZE], C21[SIZE][SIZE], C22[SIZE][SIZE];
    int M1[SIZE][SIZE], M2[SIZE][SIZE], M3[SIZE][SIZE], M4[SIZE][SIZE], M5[SIZE][SIZE], M6[SIZE][SIZE], M7[SIZE][SIZE];

    // Dividir las matrices A y B
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            A11[i][j] = A[i][j];
            A12[i][j] = A[i][j + newSize];
            A21[i][j] = A[i + newSize][j];
            A22[i][j] = A[i + newSize][j + newSize];

            B11[i][j] = B[i][j];
            B12[i][j] = B[i][j + newSize];
            B21[i][j] = B[i + newSize][j];
            B22[i][j] = B[i + newSize][j + newSize];
        }
    }

    // Cálculos de M1 a M7
    int AResult[SIZE][SIZE], BResult[SIZE][SIZE];

    // M1 = (A11 + A22) * (B11 + B22)
    addMatrix(A11, A22, AResult);
    addMatrix(B11, B22, BResult);
    multiplyMatrixStrassen(AResult, BResult, M1, newSize);

    // M2 = (A21 + A22) * B11
    addMatrix(A21, A22, AResult);
    multiplyMatrixStrassen(AResult, B11, M2, newSize);

    // M3 = A11 * (B12 - B22)
    subtractMatrix(B12, B22, BResult);
    multiplyMatrixStrassen(A11, BResult, M3, newSize);

    // M4 = A22 * (B21 - B11)
    subtractMatrix(B21, B11, BResult);
    multiplyMatrixStrassen(A22, BResult, M4, newSize);

    // M5 = (A11 + A12) * B22
    addMatrix(A11, A12, AResult);
    multiplyMatrixStrassen(AResult, B22, M5, newSize);

    // M6 = (A21 - A11) * (B11 + B12)
    subtractMatrix(A21, A11, AResult);
    addMatrix(B11, B12, BResult);
    multiplyMatrixStrassen(AResult, BResult, M6, newSize);

    // M7 = (A12 - A22) * (B21 + B22)
    subtractMatrix(A12, A22, AResult);
    addMatrix(B21, B22, BResult);
    multiplyMatrixStrassen(AResult, BResult, M7, newSize);

    // Combinar los resultados en C
    addMatrix(M1, M4, C11);
    subtractMatrix(C11, M5, C11);
    addMatrix(C11, M7, C11);

    addMatrix(M3, M5, C12);
    addMatrix(M2, M4, C21);
    subtractMatrix(M1, M2, C22);
    addMatrix(C22, M3, C22);
    addMatrix(C22, M6, C22);

    // Asignar los bloques resultantes a C
    for (int i = 0; i < newSize; ++i) {
        for (int j = 0; j < newSize; ++j) {
            C[i][j] = C11[i][j];
            C[i][j + newSize] = C12[i][j];
            C[i + newSize][j] = C21[i][j];
            C[i + newSize][j + newSize] = C22[i][j];
        }
    }
}

void printMatrix(const int matrix[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            cout << matrix[i][j] << " ";
        }
        cout << endl;
    }
}

bool compareMatrices(const int A[SIZE][SIZE], const int B[SIZE][SIZE]) {
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j)
            if (A[i][j] != B[i][j]) return false;
    return true; // Las matrices son iguales
}

int main() {
    cout << "Ejecutando para SIZE = " << SIZE << endl;

    // Generar números aleatorios entre 1 y 10
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j) {
            A[i][j] = rand() % 10 + 1;
            B[i][j] = rand() % 10 + 1;
            C[i][j] = 0; // Inicializar C a 0
        }

    XTime tProcessorStart, tProcessorEnd;

    // Multiplicación naive
    XTime_GetTime(&tProcessorStart);
    naiveMultiplication(A, B, C);
    XTime_GetTime(&tProcessorEnd);
    cout << "naive\n";
    printf("(ARM0)PS took %.5f ms. to calculate the product \n", 1000.0 * (tProcessorEnd - tProcessorStart) / (XPAR_PS7_CORTEXA9_0_CPU_CLK_FREQ_HZ));

    // Reiniciar C
    memset(C, 0, SIZE * SIZE * sizeof(int));

    // Multiplicación naive reordenada
    XTime_GetTime(&tProcessorStart);
    naiveReorderedMultiplication(A, B, C);
    XTime_GetTime(&tProcessorEnd);
    cout << "naive reordenada\n";
    printf("(ARM0)PS took %.5f ms. to calculate the product \n", 1000.0 * (tProcessorEnd - tProcessorStart) / (XPAR_PS7_CORTEXA9_0_CPU_CLK_FREQ_HZ));

    // Reiniciar C
    memset(C, 0, SIZE * SIZE * sizeof(int));

    // Multiplicación por bloques
    int blockSize = 2; // Define el tamaño del bloque
    XTime_GetTime(&tProcessorStart);
    blockMultiplication(A, B, C, blockSize);
    XTime_GetTime(&tProcessorEnd);
    cout << "bloques\n";
    printf("(ARM0)PS took %.5f ms. to calculate the product \n", 1000.0 * (tProcessorEnd - tProcessorStart) / (XPAR_PS7_CORTEXA9_0_CPU_CLK_FREQ_HZ));

    // Reiniciar C
    memset(C, 0, SIZE * SIZE * sizeof(int));

    // Multiplicación por bloques reordenada
    XTime_GetTime(&tProcessorStart);
    blockReorderedMultiplication(A, B, C, blockSize);
    XTime_GetTime(&tProcessorEnd);
    cout << "bloques reordenada\n";
    printf("(ARM0)PS took %.5f ms. to calculate the product \n", 1000.0 * (tProcessorEnd - tProcessorStart) / (XPAR_PS7_CORTEXA9_0_CPU_CLK_FREQ_HZ));

    return 0;
}
