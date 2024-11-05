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

#define SIZE 512  // Taille des matrices (512x512)

// Informations sur le timer
#define TIMER_DEVICE_ID     XPAR_XSCUTIMER_0_DEVICE_ID  // ID du timer
#define INTC_DEVICE_ID      XPAR_SCUGIC_SINGLE_DEVICE_ID // ID de l'INTC
#define TIMER_IRPT_INTR     XPAR_SCUTIMER_INTR           // Interruption du timer
#define TIMER_LOAD_VALUE    0xFFFFFFFF                   // Valeur de chargement du timer

XScuTimer Timer;  // Instance du timer

int A[SIZE][SIZE]; // Matrice A
int B[SIZE][SIZE]; // Matrice B
int C[SIZE][SIZE]; // Matrice C (produit)

// Instance du driver BRAM
XBram Bram;

// D�finitions des s�maphores pour la synchronisation
#define LOC_BRAM_SEMAPHORE 8
#define FLAG_SEMAPHORE_START_READ 0x12345678 // Indicateur pour d�marrer la lecture
#define FLAG_SEMAPHORE_STOP_READ 0x89ABCDEF  // Indicateur pour arr�ter la lecture
#define FLAG_SEMAPHORE_STOP_MULT 0x11111111   // Indicateur pour arr�ter la multiplication

// Fonction pour initialiser le p�riph�rique BRAM
int initializeDevice(u16 MutexDeviceID) {
    int Status;  // Variable pour stocker l'�tat de l'initialisation
    u16 DeviceId = MutexDeviceID; // ID du p�riph�rique BRAM
    XBram_Config *ConfigPtr = NULL; // Pointeur vers la configuration de BRAM

    // Recherche de la configuration de BRAM
    ConfigPtr = XBram_LookupConfig(DeviceId);
    if (NULL == ConfigPtr) {
        xil_printf("(ARM0) failed getting BRAM configuration\n"); // Affiche une erreur si la configuration �choue
        return XST_FAILURE;
    }

    // Initialisation du contr�leur BRAM
    Status = XBram_CfgInitialize(&Bram, ConfigPtr, ConfigPtr->CtrlBaseAddress);
    if (Status != XST_SUCCESS) {
        xil_printf("(ARM0) failed initializing BRAM controller\n"); // Affiche une erreur si l'initialisation �choue
        return XST_FAILURE;
    }

    return XST_SUCCESS; // Retourne le succ�s si tout va bien
}

// Fonction de multiplication na�ve
void naiveMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE/2; ++i) // Limite � la moiti� de SIZE pour C
        for (int j = 0; j < SIZE; ++j)
            for (int k = 0; k < SIZE; ++k)
                C[i][j] += A[i][k] * B[k][j]; // Produit de A et B
}

// Fonction de multiplication na�ve r�ordonn�e
void naiveReorderedMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE]) {
    for (int i = 0; i < SIZE/2; ++i)
        for (int k = 0; k < SIZE; ++k)
            for (int j = 0; j < SIZE; ++j)
                C[i][j] += A[i][k] * B[k][j]; // R�ordonne les boucles pour une meilleure utilisation du cache
}

// Fonction de multiplication par blocs
void blockMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE], int blockSize) {
    for (int ii = 0; ii < SIZE/2; ii += blockSize) // It�re sur les blocs
        for (int jj = 0; jj < SIZE; jj += blockSize)
            for (int kk = 0; kk < SIZE; kk += blockSize)
                for (int i = ii; i < ((ii + blockSize < SIZE) ? ii + blockSize : SIZE); ++i)
                    for (int j = jj; j < ((jj + blockSize < SIZE) ? jj + blockSize : SIZE); ++j)
                        for (int k = kk; k < ((kk + blockSize < SIZE) ? kk + blockSize : SIZE); ++k)
                            C[i][j] += A[i][k] * B[k][j]; // Produit des �l�ments par blocs
}

// Fonction de multiplication par blocs r�ordonn�e
void blockReorderedMultiplication(const int A[SIZE][SIZE], const int B[SIZE][SIZE], int C[SIZE][SIZE], int blockSize) {
    for (int ii = 0; ii < SIZE/2; ii += blockSize)
        for (int kk = 0; kk < SIZE; kk += blockSize)
            for (int jj = 0; jj < SIZE; jj += blockSize)
                for (int i = ii; i < ((ii + blockSize < SIZE) ? ii + blockSize : SIZE); ++i)
                    for (int k = kk; k < ((kk + blockSize < SIZE) ? kk + blockSize : SIZE); ++k)
                        for (int j = jj; j < ((jj + blockSize < SIZE) ? jj + blockSize : SIZE); ++j)
                            C[i][j] += A[i][k] * B[k][j]; // Produit des �l�ments en r�ordonnant les boucles
}

// Fonction principale pour la multiplication de matrices
void mat_mult_sw() {
    int dBram; // Variable pour stocker la valeur lue depuis BRAM

    XScuTimer_Config *TMRConfigPtr; // Pointeur pour la configuration du timer

    // Initialisation du timer
    TMRConfigPtr = XScuTimer_LookupConfig(TIMER_DEVICE_ID);
    XScuTimer_CfgInitialize(&Timer, TMRConfigPtr, TMRConfigPtr->BaseAddr);
    XScuTimer_SelfTest(&Timer); // Test du timer
    XScuTimer_LoadTimer(&Timer, TIMER_LOAD_VALUE); // Chargement de la valeur dans le timer

    // Remplissage des matrices A et B avec des valeurs al�atoires
    for (int i = 0; i < SIZE; ++i)
        for (int j = 0; j < SIZE; ++j) {
            A[i][j] = rand() % 10 + 1; // Remplit A avec des valeurs entre 1 et 10
            B[i][j] = rand() % 10 + 1; // Remplit B avec des valeurs entre 1 et 10
        }

    // Initialisation des dispositifs BRAM
    initializeDevice(XPAR_BRAM_0_DEVICE_ID);
    xil_printf("(ARM0) connected\n"); // Affiche que le p�riph�rique BRAM est connect�

    // �criture des matrices A et B dans BRAM
    XBram_WriteReg(XPAR_BRAM_0_BASEADDR, 0, (u32)A); // �crit A dans BRAM
    XBram_WriteReg(XPAR_BRAM_0_BASEADDR, SIZE * SIZE * sizeof(int), (u32)B); // �crit B dans BRAM

    // Envoi d'un drapeau � proc0 pour lire la matrice depuis la carte SD
    XBram_WriteReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE, FLAG_SEMAPHORE_START_READ);

    // Attente du drapeau de proc0 une fois la lecture termin�e
    dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE);
    while(dBram != FLAG_SEMAPHORE_STOP_READ) {
        dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE); // V�rifie l'�tat
    }

    // Initialisation de la matrice C � z�ro
    for(int i = 0; i < SIZE; i++) {
        for(int j = 0; j < SIZE; j++) {
            C[i][j] = 0; // Met chaque �l�ment de C � z�ro
        }
    }

    // Variables pour le chronom�trage
    XTime tProcessorStart, tProcessorEnd;
    XTime_GetTime(&tProcessorStart); // Temps de d�but

    // Multiplication de matrices
    for(int i = 0; i < SIZE/2; i++) {
        for(int k = 0; k < SIZE; k++) {
            for(int j = 0; j < SIZE; j++) {
                C[i][j] += A[i][k] * B[k][j]; // Calcule le produit
            }
        }
    }

    // Attente de la fin de la multiplication de proc0
    dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE);
    while(dBram != FLAG_SEMAPHORE_STOP_MULT) {
        dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE); // V�rifie l'�tat
    }

    // Temps de fin
    XTime_GetTime(&tProcessorEnd);

    // Calcul et affichage du temps de traitement
    printf("(ARM0)PS took %.5f ms. to calculate the product \n", 1000.0 * (tProcessorEnd - tProcessorStart) / (XPAR_PS7_CORTEXA9_0_CPU_CLK_FREQ_HZ));
    Xil_DCacheFlush(); // Vide le cache

}

// Fonction principale
int main() {
    init_platform(); // Initialise la plateforme
    xil_printf("Platform initialized 0.\n"); // Affiche que la plateforme est initialis�e
    mat_mult_sw(); // Appelle la fonction de multiplication de matrices
    cleanup_platform(); // Nettoie les ressources de la plateforme
    return 0; // Retourne 0 pour indiquer que le programme s'est termin� avec succ�s
}
