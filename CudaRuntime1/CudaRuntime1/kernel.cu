
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctime>
#include <math.h>

#define tileWidth 2

void initializeMatrix(float* Array, int SIZE) {
    // Function associates a random float at each index of the matrix
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            Array[i * SIZE + j] = (float)(rand() / RAND_MAX);

        }
    }
}

void matrixMul(float* resultMatrix, float* MatrixA, float* MatrixB, int SIZE) {

    // Initialize clock 
    clock_t startTime, stopTime;
    startTime = clock();

    // Matrix multiplication algorithm
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            resultMatrix[i * SIZE + j] = 0;

            for (int k = 0; k < SIZE; k++) {
                resultMatrix[i * SIZE + j] += MatrixA[i * SIZE + k] * MatrixB[k * SIZE + j];
            }
        }
    }

    // Stop the time and print results
    stopTime = clock();
    printf("CPU Matrix Multiplication % .2fms\n", (double)stopTime - startTime);
}


void verifyMatrix(float* CPUMatrix, float* GPUMatrix, int SIZE) {

    // For every matrix index, check if the CPU and GPU results match within an allowance of 0.01
    for (int i = 0; i < SIZE; i++) {
        for (int j = 0; j < SIZE; j++) {
            if (fabs(CPUMatrix[i * SIZE + j] - GPUMatrix[i * SIZE + j]) > 0.01) {
                printf("TEST FAILED\n");
                return;
            }
        }
    }
    // Print passed if matrices match
    printf("TEST PASSED\n");
    return;

}

__global__ void kernelMultipleMatrixMul(float* Result, float* MatrixA, float* MatrixB, int SIZE) {
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
    for (int i = 0; i < SIZE / tileWidth; i++) {
        // Load A and B shared matrices into memory
        sharedA[yThread][xThread] = MatrixA[row * SIZE + i * tileWidth + xThread];
        sharedB[yThread][xThread] = MatrixB[(i * tileWidth + yThread) * SIZE + col];
        // Ensure threads finish simultaneously 
        __syncthreads();
        // Performs dots product 
        for (int j = 0; j < tileWidth; j++) {
            temp += sharedA[yThread][j] * sharedB[j][xThread];
        }
        // Ensure threads finish simultaneously 
        __syncthreads();
        // Store result in corresponding location
        Result[row * SIZE + col] = temp;
    }

    return;
}

void gpuThreadMatrixMul(float* Result, float* MatrixA, float* MatrixB, int SIZE) {
    // Determine byte size
    int BYTES = SIZE * SIZE * sizeof(float);

    // Initialize and allocate memory to device matrices
    float* deviceMatrixA;
    float* deviceMatrixB;
    float* deviceResultMatrix;

    cudaMalloc(&deviceMatrixA, BYTES);
    cudaMalloc(&deviceMatrixB, BYTES);
    cudaMalloc(&deviceResultMatrix, BYTES);

    // Copy host matrices to the device 
    cudaMemcpy(deviceMatrixA, MatrixA, BYTES, cudaMemcpyHostToDevice);
    cudaMemcpy(deviceMatrixB, MatrixB, BYTES, cudaMemcpyHostToDevice);

    // Initialize time and CUDA events
    float time = 0;

    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaDeviceSynchronize();
  

    // Make grid and block dimensions 1 
    dim3 dimGrid(SIZE/tileWidth, SIZE/tileWidth);
    dim3 dimBlock(SIZE/tileWidth, SIZE/tileWidth);

    // Start recording and call the thread multiplication
    cudaEventRecord(startTime, 0);
    kernelMultipleMatrixMul << <dimGrid, dimBlock >> > (deviceResultMatrix, deviceMatrixA, deviceMatrixB, SIZE);

    // Stop recording and store time in appropriate variable
    cudaEventRecord(stopTime, 0);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&time, startTime, stopTime);

    // Copy the finalized matrix over to the host
    cudaMemcpy(Result, deviceResultMatrix, BYTES, cudaMemcpyDeviceToHost);

    // Print results
    printf("GPU Matrix Multiplication Time for %d size and %d tile size: %.2f\n", SIZE, tileWidth, time);

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
    int BYTES = 0;

    BYTES = 256 * 256 * sizeof(float);

    MatrixA = (float*)malloc(BYTES);
    MatrixB = (float*)malloc(BYTES);
    resultMatrix1 = (float*)malloc(BYTES);
    resultMatrix2 = (float*)malloc(BYTES);

    // Initialize matrices
    initializeMatrix(MatrixA, 256);
    initializeMatrix(MatrixB, 256);

    // Thread block sizes/ matrix sizes
    int list[5] = { 2, 4, 8, 16, 32 };
    int sizes[5] = { 256, 512, 1024, 2048, 4096 };

    // Loop through sizes
    for (int i = 0; i < 5; i++) {
        BYTES = sizes[i] * sizes[i] * sizeof(float);

        // Allocate memory for new size
        MatrixA = (float*)realloc(MatrixA, BYTES);
        MatrixB = (float*)realloc(MatrixB, BYTES);
        resultMatrix1 = (float*)realloc(resultMatrix1, BYTES);
        resultMatrix2 = (float*)realloc(resultMatrix2, BYTES);

        // Initialize larger matrices
        initializeMatrix(MatrixA, sizes[i]);
        initializeMatrix(MatrixB, sizes[i]);
        
        for (int j = 0; j < 5; j++) {
            // Call CPU matrix multiplication
            matrixMul(resultMatrix2, MatrixA, MatrixB, sizes[i]);
            // Call GPU matrix multiplications with differeing threads
            gpuThreadMatrixMul(resultMatrix1, MatrixA, MatrixB, sizes[i]);
            // Ensure the multiplcation is the same result
            verifyMatrix(resultMatrix1, resultMatrix2, sizes[i]);
        }
    }

    return 0;
}



