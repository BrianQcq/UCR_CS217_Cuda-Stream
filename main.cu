/******************************************************************************
 *cr
 *cr            (C) Copyright 2010 The Board of Trustees of the
 *cr                        University of Illinois
 *cr                         All Rights Reserved
 *cr
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.cu"

#define SegSize 512
#define StreamNum 3
#define BlockSize 512

int main (int argc, char *argv[])
{
    //set standard seed
    srand(217);

    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    float *h_A, *h_B, *h_C;
    float *d_A0, *d_B0, *d_C0;
    float *d_A1, *d_B1, *d_C1;
    float *d_A2, *d_B2, *d_C2;
    size_t A_sz, B_sz, C_sz;
    unsigned VecSize;

    cudaStream_t stream0, stream1, stream2;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    if (argc == 1) {
        VecSize = 1000000;
    } else if (argc == 2) {
        VecSize = atoi(argv[1]);   
    } else {
        printf("\nOh no!\nUsage: ./vecAdd <Size>");
        exit(0);
    }

    A_sz = VecSize;
    B_sz = VecSize;
    C_sz = VecSize;
    int leftNum = VecSize % (SegSize * StreamNum);
    cudaHostAlloc((void**)&h_A, A_sz*sizeof(float), cudaHostAllocDefault);
    for (unsigned int i=0; i < A_sz; i++) { h_A[i] = (rand()%100)/100.00; }
    cudaHostAlloc((void**)&h_B, B_sz*sizeof(float), cudaHostAllocDefault);
    for (unsigned int i=0; i < B_sz; i++) { h_B[i] = (rand()%100)/100.00; }
    cudaHostAlloc((void**)&h_C, C_sz*sizeof(float), cudaHostAllocDefault);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("size of vector: %u x 1\n  ", VecSize);
    
    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**) &d_A0, sizeof(float)*SegSize);
    cuda_ret = cudaMalloc((void**) &d_A1, sizeof(float)*SegSize);
    cuda_ret = cudaMalloc((void**) &d_A2, sizeof(float)*SegSize);
    cuda_ret = cudaMalloc((void**) &d_B0, sizeof(float)*SegSize);
    cuda_ret = cudaMalloc((void**) &d_B1, sizeof(float)*SegSize);
    cuda_ret = cudaMalloc((void**) &d_B2, sizeof(float)*SegSize);
    cuda_ret = cudaMalloc((void**) &d_C0, sizeof(float)*SegSize);
    cuda_ret = cudaMalloc((void**) &d_C1, sizeof(float)*SegSize);
    cuda_ret = cudaMalloc((void**) &d_C2, sizeof(float)*SegSize);
    if (cuda_ret != cudaSuccess) {
        printf("Fail to cudaMalloc on GPU");
        exit(EXIT_FAILURE);
    }
    
    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    
    // Launch kernel  ---------------------------
    printf("Launching kernel..."); fflush(stdout);
    startTime(&timer);
    int i;
    for(i = 0; i < VecSize; i += SegSize * StreamNum)
    {
        cudaMemcpyAsync(d_A0, h_A + i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_B0, h_B + i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_A1, h_A + i + SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_B1, h_B + i + SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_A2, h_A + i + 2 * SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_B2, h_B + i + 2 * SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream2);

        VecAdd<<<SegSize / BlockSize, BlockSize, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
        VecAdd<<<SegSize / BlockSize, BlockSize, 0, stream1>>>(d_A1, d_B1, d_C1, SegSize);
        VecAdd<<<SegSize / BlockSize, BlockSize, 0, stream2>>>(d_A2, d_B2, d_C2, SegSize);

        cudaMemcpyAsync(h_C + i, d_C0, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_C + i + SegSize, d_C1, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(h_C + i + 2 * SegSize, d_C2, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    }

    // deal with the left data

    i -= SegSize * StreamNum;
    if(leftNum > 2 * SegSize)
    {
        cudaMemcpyAsync(d_A0, h_A + i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_B0, h_B + i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_A1, h_A + i + SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_B1, h_B + i + SegSize, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_A2, h_A + i + 2 * SegSize, (leftNum - 2 * SegSize) * sizeof(float), cudaMemcpyHostToDevice, stream2);
        cudaMemcpyAsync(d_B2, h_B + i + 2 * SegSize, (leftNum - 2 * SegSize) * sizeof(float), cudaMemcpyHostToDevice, stream2);

        VecAdd<<<1, BlockSize, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
        VecAdd<<<1, BlockSize, 0, stream1>>>(d_A1, d_B1, d_C1, SegSize);
        VecAdd<<<1, leftNum - 2 * SegSize, 0, stream2>>>(d_A2, d_B2, d_C2, leftNum - 2 * SegSize);

        cudaMemcpyAsync(h_C + i, d_C0, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_C + i + SegSize, d_C1, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream1);
        cudaMemcpyAsync(h_C + i + 2 * SegSize, d_C2, (leftNum - 2 * SegSize) * sizeof(float), cudaMemcpyDeviceToHost, stream2);
    }
    else if(leftNum > SegSize && leftNum <= 2*SegSize)
    {
        cudaMemcpyAsync(d_A0, h_A + i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_B0, h_B + i, SegSize * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_A1, h_A + i + SegSize, (leftNum - SegSize) * sizeof(float), cudaMemcpyHostToDevice, stream1);
        cudaMemcpyAsync(d_B1, h_B + i + SegSize, (leftNum - SegSize) * sizeof(float), cudaMemcpyHostToDevice, stream1);

        VecAdd<<<1, BlockSize, 0, stream0>>>(d_A0, d_B0, d_C0, SegSize);
        VecAdd<<<1, leftNum - SegSize, 0, stream1>>>(d_A1, d_B1, d_C1, leftNum - SegSize);

        cudaMemcpyAsync(h_C + i, d_C0, SegSize * sizeof(float), cudaMemcpyDeviceToHost, stream0);
        cudaMemcpyAsync(h_C + i + SegSize, d_C1, (leftNum - SegSize) * sizeof(float), cudaMemcpyDeviceToHost, stream1);
    }
    else if(leftNum > 0 && leftNum <= SegSize)
    {
        cudaMemcpyAsync(d_A0, h_A + i, leftNum * sizeof(float), cudaMemcpyHostToDevice, stream0);
        cudaMemcpyAsync(d_B0, h_B + i, leftNum * sizeof(float), cudaMemcpyHostToDevice, stream0);

        VecAdd<<<1, leftNum, 0, stream0>>>(d_A0, d_B0, d_C0, leftNum);

        cudaMemcpyAsync(h_C + i, d_C0, leftNum * sizeof(float), cudaMemcpyDeviceToHost, stream0);
    }
    cudaStreamSynchronize(stream0);
    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);
    cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to launch kernel");
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Verify correctness -----------------------------------------------------

    printf("Verifying results..."); fflush(stdout);

    verify(h_A, h_B, h_C, VecSize);

    // Free memory ------------------------------------------------------------

    cudaFree(d_A0);
    cudaFree(d_A1);
    cudaFree(d_A2);
    cudaFree(d_B0);
    cudaFree(d_B1);
    cudaFree(d_B2);
    cudaFree(d_C0);
    cudaFree(d_C1);
    cudaFree(d_C2);
    
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C);

    return 0;
}
