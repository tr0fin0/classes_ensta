/******************************************************************************
* code dual core, core 1
*******************************************************************************/

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

#define SIZE 1024

static char INPUT_FILE_A[32] = "A1024.bin";
static char INPUT_FILE_B[32] = "B1024.bin";
static char INPUT_FILE_C_golden[32] = "C1024.bin";
static char *Log_File;

static FIL file1,file2,file3;

//timer info
#define TIMER_DEVICE_ID		XPAR_XSCUTIMER_0_DEVICE_ID
#define INTC_DEVICE_ID		XPAR_SCUGIC_SINGLE_DEVICE_ID
#define TIMER_IRPT_INTR		XPAR_SCUTIMER_INTR
#define TIMER_LOAD_VALUE	0xFFFFFFFF

XScuTimer	Timer;


XBram Bram;	/* The Instance of the BRAM Driver */

#define LOC_BRAM_SEMAPHORE 8
#define FLAG_SEMAPHORE_START_READ 0x12345678
#define FLAG_SEMAPHORE_STOP_READ 0x89ABCDEF
#define FLAG_SEMAPHORE_STOP_MULT 0x11111111


#define LED_DELAY 40000000


double mat_mult_sw(){
	int dBram;

	FRESULT f_inA, f_inB, f_inC;

	// Wait the flag from proc0 to initialize the reading of the matrix from SD card
	dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE);
	while(dBram != FLAG_SEMAPHORE_START_READ){
		dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE);
	}

	static FATFS  FS_instance;
	const char *Path = "0:/";
	FRESULT  result;
	result = f_mount(&FS_instance,Path, 0);
	if (result != FR_OK) {
		xil_printf("(ARM1)Cannot mount sd\n");
		return XST_FAILURE;
	}

	// Read matrices from SD card
	int A[SIZE][SIZE]__attribute__((aligned(32)));
	int B[SIZE][SIZE]__attribute__((aligned(32)));
	int C[SIZE][SIZE]__attribute__((aligned(32)));
	int C_golden[SIZE][SIZE]__attribute__((aligned(32)));

	// Open the input, output, golden files, read the input and golden
	// and store them to the corresponding arrays.
	Log_File = (char *)INPUT_FILE_A;
	f_inA = f_open(&file1, Log_File,FA_READ);
	xil_printf("(ARM1)ERROR:%d\n",f_inA);
	if (f_inA!= FR_OK) {
		xil_printf("(ARM1)File matrix A not found\n");
		return XST_FAILURE;
	}
	Log_File = (char *)INPUT_FILE_B;
	f_inB = f_open(&file2, Log_File,FA_READ);
	if (f_inB!= FR_OK) {
		xil_printf("(ARM1)File matrix B not found\n");
		return XST_FAILURE;
	}
	Log_File = (char *)INPUT_FILE_C_golden;
	f_inC = f_open(&file3, Log_File,FA_READ);
	if (f_inC!= FR_OK) {
		xil_printf("(ARM1)File matrix C not found\n");
		return XST_FAILURE;
	}

	// Read A
	uint readBytes=0;
	f_read(&file1,&A[0],SIZE*SIZE*sizeof(int),&readBytes);
	if(readBytes == sizeof(int)*SIZE*SIZE)
		xil_printf("(ARM1)Read the complete data for A\n");
	else
		xil_printf("(ARM1)Failed reading A, read %d bytes\n",readBytes);

	// Read B
	readBytes=0;
	f_read(&file2,&B[0],SIZE*SIZE*sizeof(int),&readBytes);
	if(readBytes == sizeof(int)*SIZE*SIZE)
		xil_printf("(ARM1)Read the complete data for B\n");
	else
		xil_printf("(ARM1)Failed reading B, read %d bytes\n",readBytes);

	// Read C_golden
	readBytes=0;
	f_read(&file3,&C_golden[0],SIZE*SIZE*sizeof(int),&readBytes);
	if(readBytes == sizeof(int)*SIZE*SIZE)
		xil_printf("(ARM1)Read the complete data for C\n");
	else
		xil_printf("(ARM1)Failed reading C, read %d bytes\n",readBytes);

	f_close(&file1);
	f_close(&file2);
	f_close(&file3);

	Xil_DCacheFlush();

	// Initialize matrix C
	for(int i = 0; i < SIZE; i++){
		for(int j = 0; j < SIZE; j++){
			C[i][j] = 0;
		}
	}

	xil_printf("(ARM1) Start matrix multiplication.\n");

	// Send flag to proc0 when proc1 finishes to read the matrix from SD card
	XBram_WriteReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE, FLAG_SEMAPHORE_STOP_READ);

	// Calculate half of matrix multiplication
//	for(int i = SIZE/2; i < SIZE; i++){
//		for(int j = 0; j < SIZE; j++){
//			for(int k = 0; k < SIZE; k++){
//				C[i][j] += A[i][k]*B[k][j];
//			}
//		}
//	}

	// Better use the cache by changing the order of the loops
	for(int i = SIZE/2; i < SIZE; i++){
		for(int k = 0; k < SIZE; k++){
			for(int j = 0; j < SIZE; j++){
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}

	// Send flag to proc0 when proc1 finishes the its part of matrix multiplication
	XBram_WriteReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE, FLAG_SEMAPHORE_STOP_MULT);

	xil_printf("(ARM1) Finish matrix multiplication.\n");

	// Verify output
	int success = 1;
	for(int i = SIZE/2; i < SIZE; i++){
		for(int j = 0; j < SIZE; j++){
			if(C[i][j] != C_golden[i][j]){
				success = 0;
				xil_printf("i=%d|j=%d|C_ij=%d|C_g_ij=%d\n",i,j,C[i][j],C_golden[i][j]);
			}
		}
	}


	return success;
}



int main()
{

	init_platform();
	xil_printf("(ARM1) Hi there\n");

	int success = mat_mult_sw();
	if(success){
		xil_printf("Success");
	}else{
		xil_printf("Failed");
	}


	xil_printf("(ARM1) This is the end\n\r");

    cleanup_platform();
    return 0;
}
