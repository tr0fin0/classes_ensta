#include <stdio.h>
#include "platform.h"
#include "xil_io.h"
#include "xparameters.h"
#include "xil_printf.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>
#include "xsdps.h"
#include "ff.h"
#include "xil_types.h"
#include "xscutimer.h"
#include <sys/time.h>
#include "xtime_l.h"
#include "xbram.h"

#define SIZE 512 // D�finition de la taille des matrices (512x512)

// Informations sur le timer
#define TIMER_DEVICE_ID         XPAR_XSCUTIMER_0_DEVICE_ID // ID du timer
#define INTC_DEVICE_ID          XPAR_SCUGIC_SINGLE_DEVICE_ID // ID de l'INTC
#define TIMER_IRPT_INTR         XPAR_SCUTIMER_INTR // Interruption du timer
#define TIMER_LOAD_VALUE        0xFFFFFFFF // Valeur � charger dans le timer

XScuTimer Timer; // Instance du timer
XBram Bram; // Instance du driver BRAM

// D�finitions des s�maphores pour la synchronisation entre les processeurs
#define LOC_BRAM_SEMAPHORE 8
#define FLAG_SEMAPHORE_START_READ 0x12345678 // Indicateur pour d�marrer la lecture
#define FLAG_SEMAPHORE_STOP_READ 0x89ABCDEF // Indicateur pour arr�ter la lecture
#define FLAG_SEMAPHORE_STOP_MULT 0x11111111 // Indicateur pour arr�ter la multiplication

// Matrices globales pour stocker les valeurs
int A[SIZE][SIZE]; // Matrice A
int B[SIZE][SIZE]; // Matrice B
int C[SIZE][SIZE]; // Matrice C (produit)

// Fonction de multiplication na�ve
void naiveMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE]) {
    // It�re sur la moiti� de la matrice A pour effectuer la multiplication
    for (int i = SIZE / 2; i < SIZE; ++i) // Limite � la moiti� de SIZE
        for (int j = 0; j < SIZE; ++j)
            for (int k = 0; k < SIZE; ++k)
                C[i][j] += A[i][k] * B[k][j]; // Calcule le produit et l'ajoute � C
}

// Fonction de multiplication na�ve r�ordonn�e
void naiveReorderedMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE]) {
    // R�ordonne les boucles pour une meilleure exploitation du cache
    for (int i = SIZE / 2; i < SIZE; ++i)
        for (int k = 0; k < SIZE; ++k)
            for (int j = 0; j < SIZE; ++j)
                C[i][j] += A[i][k] * B[k][j]; // Calcule le produit et l'ajoute � C
}

// Fonction de multiplication par blocs
void blockMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE], int blockSize) {
    // It�ration par blocs pour am�liorer la performance
    for (int ii = 0; ii < SIZE / 2; ii += blockSize) // It�ration sur les blocs de lignes
        for (int jj = 0; jj < SIZE; jj += blockSize) // It�ration sur les blocs de colonnes
            for (int kk = 0; kk < SIZE; kk += blockSize) // It�ration sur les blocs interm�diaires
                for (int i = ii; i < ((ii + blockSize < SIZE) ? ii + blockSize : SIZE); ++i)
                    for (int j = jj; j < ((jj + blockSize < SIZE) ? jj + blockSize : SIZE); ++j)
                        for (int k = kk; k < ((kk + blockSize < SIZE) ? kk + blockSize : SIZE); ++k)
                            C[i][j] += A[i][k] * B[k][j]; // Produit des �l�ments par blocs
}

// Fonction de multiplication par blocs r�ordonn�e
void blockReorderedMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE], int blockSize) {
    // R�ordonne les boucles pour une meilleure exploitation du cache lors de la multiplication par blocs
    for (int ii = 0; ii < SIZE / 2; ii += blockSize)
        for (int kk = 0; kk < SIZE; kk += blockSize)
            for (int jj = 0; jj < SIZE; jj += blockSize)
                for (int i = ii; i < ((ii + blockSize < SIZE) ? ii + blockSize : SIZE); ++i)
                    for (int k = kk; k < ((kk + blockSize < SIZE) ? kk + blockSize : SIZE); ++k)
                        for (int j = jj; j < ((jj + blockSize < SIZE) ? jj + blockSize : SIZE); ++j)
                            C[i][j] += A[i][k] * B[k][j]; // Produit des �l�ments par blocs
}

// Fonction principale pour la multiplication de matrices
void mat_mult_sw() {
    int dBram; // Variable pour stocker la valeur lue depuis BRAM

    // Attendre le drapeau de proc0 pour initialiser la lecture des matrices depuis la carte SD
    dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE);
    while(dBram != FLAG_SEMAPHORE_START_READ) { // Boucle jusqu'� ce que le drapeau soit re�u
        dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE); // V�rifie l'�tat du s�maphore
    }

    // Lire les matrices A et B depuis BRAM
    memcpy(A, (int *)XPAR_BRAM_0_BASEADDR, SIZE * SIZE * sizeof(int)); // Copie A depuis BRAM
    memcpy(B, (int *)(XPAR_BRAM_0_BASEADDR + SIZE * SIZE * sizeof(int)), SIZE * SIZE * sizeof(int)); // Copie B depuis BRAM

    Xil_DCacheFlush(); // Vide le cache pour garantir la coh�rence des donn�es

    // Initialisation de la matrice C � z�ro
    for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            C[i][j] = 0; // Met chaque �l�ment de C � z�ro
        }
    }

    // Envoi d'un drapeau � proc0 lorsque proc1 a fini de lire les matrices depuis BRAM
    XBram_WriteReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE, FLAG_SEMAPHORE_STOP_READ);

    // Attente du drapeau de proc0 pour commencer la multiplication des matrices
    // Calcule la moiti� de la multiplication des matrices
    naiveMultiplication(A, B, C); // Appelle la fonction de multiplication na�ve

    // Envoi d'un drapeau � proc0 lorsque proc1 a fini sa partie de multiplication de matrices
    XBram_WriteReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE, FLAG_SEMAPHORE_STOP_MULT);

    xil_printf("(ARM1) Finish matrix multiplication.\n"); // Affiche que la multiplication est termin�e
}

// Fonction principale
int main() {
    init_platform(); // Initialise la plateforme
    xil_printf("Platform initialized 1.\n"); // Affiche que la plateforme est initialis�e
    mat_mult_sw(); // Appelle la fonction de multiplication de matrices
    cleanup_platform(); // Nettoie les ressources de la plateforme
    return 0; // Retourne 0 pour indiquer que le programme s'est termin� avec succ�s
}
