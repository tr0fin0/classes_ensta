#include "mpi.h"
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <assert.h>

using namespace std;


int main( int argc, char* argv[] )
{
    double startTime, endTime;
    char processorName[MPI_MAX_PROCESSOR_NAME];
    int processId;
    int nameLen;
    int numProcesses;

    // Initiating parallel MPI for process below
    MPI_Status stat;
    MPI::Init();

    MPI_Comm_size(MPI_COMM_WORLD, &numProcesses);   // store's Comm_size
    MPI_Get_processor_name(processorName, &nameLen);// store's name and it's size

    processId = MPI::COMM_WORLD.Get_rank();
    unsigned int receivedElement;

    if(processId == 0) {
        // if it is the main process 
        // --- We are getting info from text file


        // int SIZE;
        // unsigned int value;
        // ifstream fd("testing_ingo.txt");
        // fd >> SIZE; // the first number on the first line is the number of numbers in file
        // unsigned int *array = new unsigned int [SIZE];

        // for(int c = 0 ; c < SIZE ; c++)  {
        //     fd >> value;
        //     array = value;
        // }
        // --- DEBUG: in order to check if information was read properly
        //for(int c = 0; c < SIZE; c++) {
        //  printf("   %d   ", array);
        //  if((c+1)%5 == 0 )
        //      printf("\n",SIZE);
        //}


        // --- initialize data to be sorted
        int SIZE = 10;
        int randRange = SIZE;
        // int randRange = 1000;
        int array[SIZE];

        for(int i = 0; i < SIZE; i++) {
            array[i]=rand()%randRange;
        }

        std::cout << array << std::endl;


        // starting time calculation of the sort
        startTime = MPI_Wtime();

        // min and max values are got, limits of each bucket
        unsigned int min = array[0];
        unsigned int max = array[0];
        for(int i = 0; i < SIZE; i++) {
            if((unsigned)array[i] < min) { min = array[i]; }
            if((unsigned)array[i] > max) { max = array[i]; }
        }

        // calculating how many numbers each bucket/process will get
        int *numElementsArray = new int[numProcesses];

        // d skips 0, process 0 is main process
        for(int d = 1; d < numProcesses; d++) {
            numElementsArray[d] = 0;
        }

        for(int d = 0; d < SIZE; d++) {
            // int increaseOf;

            // if( (numProcesses - 1) == 0 ) {
            //     std::cout << "divide by zero master" << std::endl;
            // } else {
            //     // int increaseOf = max/(numProcesses-1);
            //     increaseOf = max/(numProcesses-1);
            // }

            int increaseOf = max/(numProcesses-1);
            int iteration = 1;
            bool pridetas = false;
            
            for(unsigned int j = increaseOf; j <= max; j = j + increaseOf) {
                if((unsigned)array[d] <= j) {
                    numElementsArray[iteration]++;
                    pridetas = true;
                    break;
                }
                iteration++;
            }

            if (!pridetas) { numElementsArray[iteration-1]++; }
        }

        // Sending how many each process/bucket will get numbers
        for(int i=1; i<numProcesses; i++) {
            MPI_Send(&numElementsArray[i], 1, MPI_INT, i, 2, MPI_COMM_WORLD);
        }

        // doing the same, this time sending the numbers
        for(int d=0; d < SIZE; d++) {
            // int increaseOf;

            // if( (numProcesses - 1) == 0 ) {
            //     std::cout << "divide by zero slave" << std::endl;
            // } else {
            //     // int increaseOf = max/(numProcesses-1);
            //     increaseOf = max/(numProcesses-1);
            // }

            int increaseOf = max/(numProcesses-1);
            int iteration = 1;
            bool issiunte = false;
            for (unsigned int j = increaseOf; j <= max; j = j + increaseOf) {
                if((unsigned)array[d] <= j) {
                    MPI_Send(&array[d], 1, MPI_UNSIGNED, iteration, 6, MPI_COMM_WORLD);
                    issiunte = true;
                    break;
                }
                iteration++;
            }

            if (!issiunte) {
                MPI_Send(&array[d], 1, MPI_UNSIGNED, iteration-1, 4, MPI_COMM_WORLD);

            }
        }

        // Getting back results and adding them to one array
        int lastIndex = 0; int interationIndex = 0;
        for(int i=1; i < numProcesses; i++) {
            unsigned int * recvArray = new unsigned int [numElementsArray[i]];
            MPI_Recv(&recvArray[0], numElementsArray[i], MPI_UNSIGNED, i, 1000, MPI_COMM_WORLD, &stat);
            if(lastIndex == 0) {
                lastIndex = numElementsArray[i];
            }
            for(int j=0; j<numElementsArray[i]; j++) {
                array[interationIndex] = recvArray[j];
                interationIndex++;
            }
        }

        // stoping the time
        endTime   = MPI_Wtime();

        // // showing results in file
        // ofstream fr("results.txt");
        // for(int c = 0 ; c < SIZE ; c++)  {
        //     fr << array << endl;
        // }
        // fr.close();
        // printf("Numbers: %d \n", SIZE);

        // sorting results
        printf("time: %f s\n", endTime-startTime);
        printf("process: %d\n", numProcesses);

    //----------------------------------------------------------------------------------------------------------------
    } else {
        // if child process
        int elementQtyUsed;
        // --- getting the number of numbers in the bucket
        MPI_Recv(&elementQtyUsed, 1, MPI_INT, 0, -2, MPI_COMM_WORLD, &stat);


        unsigned int *localArray = new unsigned int [elementQtyUsed]; // initiating a local bucket

        // --- getting numbers from the main process
        for(int li = 0; li < elementQtyUsed; li++) {
            MPI_Recv(&receivedElement, 1, MPI_UNSIGNED, 0, -4, MPI_COMM_WORLD, &stat);
            localArray[li] =  receivedElement;

            // --- sorting the bucket
            sort(localArray, localArray+elementQtyUsed);
            // --- sending back sorted array
            MPI_Send(localArray, elementQtyUsed, MPI_UNSIGNED, 0, 1000, MPI_COMM_WORLD);
        }

        MPI::Finalize();
        return 0;
    }
}