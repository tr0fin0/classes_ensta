/******************************************************************************
* code dual core, core 0
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

int initializeDevice(u16 MutexDeviceID){
	int Status;
	u16 DeviceId = MutexDeviceID;
	XBram_Config *ConfigPtr = NULL;

	ConfigPtr = XBram_LookupConfig(DeviceId);
	if (NULL == ConfigPtr)
	{
		xil_printf("(ARM0) failed getting BRAM configuration\n");
		return XST_FAILURE;
	}

	Status = XBram_CfgInitialize(&Bram, ConfigPtr, ConfigPtr->CtrlBaseAddress);
	if (Status != XST_SUCCESS) {
		xil_printf("(ARM0) failed initializing BRAM controller\n");
		return XST_FAILURE;
	}

	return XST_SUCCESS;
}

double mat_mult_sw(){
	static FATFS  FS_instance;
	const char *Path = "0:/";
	FRESULT  result;
	int dBram;

	FRESULT f_inA, f_inB, f_inC;

	XScuTimer_Config *TMRConfigPtr;     //timer config

	TMRConfigPtr = XScuTimer_LookupConfig(TIMER_DEVICE_ID);
	XScuTimer_CfgInitialize(&Timer, TMRConfigPtr,TMRConfigPtr->BaseAddr);
	XScuTimer_SelfTest(&Timer);
	//load the timer
	XScuTimer_LoadTimer(&Timer, TIMER_LOAD_VALUE);

	int A[SIZE][SIZE]__attribute__((aligned(32)));
	int B[SIZE][SIZE]__attribute__((aligned(32)));
	int C[SIZE][SIZE]__attribute__((aligned(32)));
	int C_golden[SIZE][SIZE]__attribute__((aligned(32)));

	result = f_mount(&FS_instance,Path, 0);
	if (result != FR_OK) {
		xil_printf("(ARM0)Cannot mount sd\n");
		return XST_FAILURE;
	}

	// Initialize BRAM devices
	initializeDevice(XPAR_BRAM_0_DEVICE_ID);
	xil_printf("(ARM0) connected\n");

	// Open the input, output, golden files, read the input and golden
	// and store them to the corresponding arrays.
	Log_File = (char *)INPUT_FILE_A;
	f_inA = f_open(&file1, Log_File,FA_READ);
	xil_printf("(ARM0)ERROR:%d\n",f_inA);
	if (f_inA!= FR_OK) {
		xil_printf("(ARM0)File matrix A not found\n");
		return XST_FAILURE;
	}
	Log_File = (char *)INPUT_FILE_B;
	f_inB = f_open(&file2, Log_File,FA_READ);
	if (f_inB!= FR_OK) {
		xil_printf("(ARM0)File matrix B not found\n");
		return XST_FAILURE;
	}
	Log_File = (char *)INPUT_FILE_C_golden;
	f_inC = f_open(&file3, Log_File,FA_READ);
	if (f_inC!= FR_OK) {
		xil_printf("(ARM0)File matrix C not found\n");
		return XST_FAILURE;
	}

	// Read A
	uint readBytes=0;
	f_read(&file1,&A[0],SIZE*SIZE*sizeof(int),&readBytes);
	if(readBytes == sizeof(int)*SIZE*SIZE)
		xil_printf("(ARM0)Read the complete data for A\n");
	else
		xil_printf("(ARM0)Failed reading A, read %d bytes\n",readBytes);

	// Read B
	readBytes=0;
	f_read(&file2,&B[0],SIZE*SIZE*sizeof(int),&readBytes);
	if(readBytes == sizeof(int)*SIZE*SIZE)
		xil_printf("(ARM0)Read the complete data for B\n");
	else
		xil_printf("(ARM0)Failed reading B, read %d bytes\n",readBytes);

	// Read C_golden
	readBytes=0;
	f_read(&file3,&C_golden[0],SIZE*SIZE*sizeof(int),&readBytes);
	if(readBytes == sizeof(int)*SIZE*SIZE)
		xil_printf("(ARM0)Read the complete data for C\n");
	else
		xil_printf("(ARM0)Failed reading C, read %d bytes\n",readBytes);

	f_close(&file1);
	f_close(&file2);
	f_close(&file3);
	f_unmount(Path);

	Xil_DCacheFlush();

	// Send flag to proc0 to read the matrix from SD card
	XBram_WriteReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE, FLAG_SEMAPHORE_START_READ);

	// Wait the flag from proc0 when proc0 finished to read the matrix from SD card
	dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE);
	while(dBram != FLAG_SEMAPHORE_STOP_READ){
		dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE);
	}

	// Initialize matrix C
	for(int i = 0; i < SIZE; i++){
		for(int j = 0; j < SIZE; j++){
			C[i][j] = 0;
		}
	}

	xil_printf("(ARM0) Start matrix multiplication.\n");
	xil_printf("(ARM0)Start timer\r\n");
	XTime tProcessorStart, tProcessorEnd;
	XTime_GetTime(&tProcessorStart);

	// Calculate half of matrix multiplication
//	for(int i = 0; i < SIZE/2; i++){
//		for(int j = 0; j < SIZE; j++){
//			for(int k = 0; k < SIZE; k++){
//				C[i][j] += A[i][k]*B[k][j];
//			}
//		}
//	}

	// Better use the cache by changing the order of the loops
	for(int i = 0; i < SIZE/2; i++){
		for(int k = 0; k < SIZE; k++){
			for(int j = 0; j < SIZE; j++){
				C[i][j] += A[i][k]*B[k][j];
			}
		}
	}

	// Wait the proc0 to finish its part of matrix multiplication
	dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE);
	while(dBram != FLAG_SEMAPHORE_STOP_MULT){
		dBram = XBram_ReadReg(XPAR_BRAM_0_BASEADDR, LOC_BRAM_SEMAPHORE);
	}

	XTime_GetTime(&tProcessorEnd);
	xil_printf("(ARM0)Measure timer\r\n");
	xil_printf("(ARM0) Finish matrix multiplication.\n");

	// Calculate the time
	printf("(ARM0)PS took %.5f ms. to calculate the product \n", 1000.0 * (tProcessorEnd - tProcessorStart) / (XPAR_PS7_CORTEXA9_0_CPU_CLK_FREQ_HZ));

	Xil_DCacheFlush();

	// Verify output
	int success = 1;
	for(int i = 0; i < SIZE/2; i++){
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
	xil_printf("(ARM0) Hi there\n");

	int success = mat_mult_sw();

	if(success){
		xil_printf("Success");
	}else{
		xil_printf("Failed");
	}

	xil_printf("(ARM0) This is the end\n\r");

    cleanup_platform();
    return 0;
}
