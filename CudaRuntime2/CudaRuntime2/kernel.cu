
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <math.h>

#define tileWidth 2

void initializeMatrix(float* Array, int dimOne, int dimTwo) {
    // Function associates a random float at each index of the matrix
    for (int i = 0; i < dimOne; i++) {
        for (int j = 0; j < dimTwo; j++) {
            Array[i * dimTwo + j] = (float)(rand() / RAND_MAX);

        }
    }
}

void matrixMul(float* resultMatrix, float* MatrixA, float* MatrixB, int dimOne, int dimTwo, int dimThree) {
    // Initialize clock 
    clock_t startTime, stopTime;
    startTime = clock();

    // Basic Matrix Multiplication Algorithm
    for (int i = 0; i < dimOne; i++) {
        for (int j = 0; j < dimThree; j++) {
            float matrixSum = 0;
            for (int k = 0; k < dimTwo; k++) {
                matrixSum += MatrixA[i * dimTwo + k] * MatrixB[k * dimThree + j];
            }
            resultMatrix[i * dimThree + j] = matrixSum;
        }
    }
    // Stop the time and print results
    stopTime = clock();
    printf("CPU Matrix Multiplication % .2fms\n", (double)stopTime - startTime);

    return;
}



void verifyMatrix(float* CPUMatrix, float* GPUMatrix, int dimOne, int dimThree) {

    // For every matrix index, check if the CPU and GPU results match within an allowance of 0.01
    for (int i = 0; i < dimOne; i++) {
        for (int j = 0; j < dimThree; j++) {
            if (fabs(CPUMatrix[i * dimThree + j] - GPUMatrix[i * dimThree + j]) > 0.01) {
                printf("TEST FAILED\n");
                return;
            }
        }
    }
    // Print passed if matrices match
    printf("TEST PASSED\n");
    return;

}

__global__ void kernelMultipleMatrixMul(float* Result, float* MatrixA, float* MatrixB, int dimOne, int dimTwo, int dimThree) {
    // Calculate thread row and columns 

    // Create shared memory for each matrix's tile
    __shared__ float sharedA[tileWidth][tileWidth];
    __shared__ float sharedB[tileWidth][tileWidth];

    int xBlock = blockIdx.x;
    int yBlock = blockIdx.y;

    int xThread = threadIdx.x;
    int yThread = threadIdx.y;

    // Determine row and col based on based on corresponding values to the result matrix
    int row = yBlock * tileWidth + yThread;
    int col = xBlock * tileWidth + xThread;

    float temp = 0;
    // Matrix multiplication with threads

    // Loop through all phases for the dot product of a tile
    for (int i = 0; i < (dimTwo + tileWidth -1) / tileWidth; i++) {
        // Load A and B shared matrices into memory
        sharedA[yThread][xThread] = MatrixA[row * dimTwo + i * tileWidth + xThread];
        sharedB[yThread][xThread] = MatrixB[(i * tileWidth + yThread) * dimThree + col];
        // Ensure threads finish simultaneously 
        __syncthreads();
        // Performs dots product 
        for (int j = 0; j < tileWidth; j++) {
            temp += sharedA[yThread][j] * sharedB[j][xThread];
        }
        // Ensure threads finish simultaneously 
        __syncthreads();
        // Store result in corresponding location
    }
    Result[row * dimThree + col] = temp;

    return;
}

void gpuThreadMatrixMul(float* Result, float* MatrixA, float* MatrixB, int dimOne, int dimTwo, int dimThree) {
    // Determine byte size
    int BYTESA = dimOne * dimTwo * sizeof(float);
    int BYTESB = dimThree * dimTwo * sizeof(float);
    int BYTESC = dimOne * dimThree * sizeof(float);


    // Initialize and allocate memory to device matrices
    float* deviceMatrixA;
    float* deviceMatrixB;
    float* deviceResultMatrix;

    cudaMalloc(&deviceMatrixA, BYTESA);
    cudaMalloc(&deviceMatrixB, BYTESB);
    cudaMalloc(&deviceResultMatrix, BYTESC);

    // Copy host matrices to the device 
    cudaMemcpy(deviceMatrixA, MatrixA, BYTESA, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, MatrixB, BYTESB, cudaMemcpyHostToDevice);

    // Initialize time and CUDA events
    float time = 0;

    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaDeviceSynchronize();


    // Make grid and block dimensions 1 
    dim3 dimGrid((dimThree + tileWidth -1)/ tileWidth, (dimOne + tileWidth -1) / tileWidth);
    dim3 dimBlock(tileWidth, tileWidth);

    // Start recording and call the thread multiplication
    cudaEventRecord(startTime, 0);
    kernelMultipleMatrixMul << <dimGrid, dimBlock >> > (deviceResultMatrix, deviceMatrixA, deviceMatrixB, dimOne, dimTwo, dimThree);

    // Stop recording and store time in appropriate variable
    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&time, startTime, stopTime);

    // Copy the finalized matrix over to the host
    cudaMemcpy(Result, deviceResultMatrix, BYTESC, cudaMemcpyDeviceToHost);

    // Print results
    printf("GPU Matrix Multiplication Time for %dx%d and %dx%d size and %d tile size: %.2f\n", dimOne, dimTwo, dimTwo, dimThree, tileWidth, time);

    // Free memory and events
    cudaEventDestroy(startTime);
    cudaEventDestroy(stopTime);
    cudaFree(deviceMatrixA);
    cudaFree(deviceMatrixB);
    cudaFree(deviceResultMatrix);
    cudaDeviceReset();
}

int main()
{

    // Initialize all matrices
    float* MatrixA;
    float* MatrixB;
    float* resultMatrix1;
    float* resultMatrix2;

    // Set matrices to first memory size

    int BYTES = 750 * 800 * sizeof(float);

    MatrixA = (float*)malloc(750 * 800 * sizeof(float));
    MatrixB = (float*)malloc(850 * 800 * sizeof(float));
    resultMatrix1 = (float*)malloc(750 * 800 * sizeof(float));
    resultMatrix2 = (float*)malloc(750 * 800 * sizeof(float));

    // Initialize matrices
    initializeMatrix(MatrixA, 750, 800);
    initializeMatrix(MatrixB, 800, 850);

    // Thread block sizes/ matrix sizes
    //int sizes[5] = { 256, 512, 1024, 2048, 4096 };

    // Loop through sizes
    //for (int i = 0; i < 5; i++) {
        //BYTES = sizes[i] * sizes[i] * sizeof(float);

        // Allocate memory for new size
        //MatrixA = (float*)realloc(MatrixA, BYTES);
        //MatrixB = (float*)realloc(MatrixB, BYTES);
        //resultMatrix1 = (float*)realloc(resultMatrix1, BYTES);
        //resultMatrix2 = (float*)realloc(resultMatrix2, BYTES);

        // Initialize larger matrices
        //initializeMatrix(MatrixA, sizes[i]);
        //initializeMatrix(MatrixB, sizes[i]);

        // Call CPU matrix multiplication
        matrixMul(resultMatrix2, MatrixA, MatrixB, 750, 800, 850);
        // Call GPU matrix multiplications with differeing threads
        gpuThreadMatrixMul(resultMatrix1, MatrixA, MatrixB, 750, 800, 850);
        // Ensure the multiplcation is the same result
        verifyMatrix(resultMatrix1, resultMatrix2, 750, 850);


    return 0;
}



