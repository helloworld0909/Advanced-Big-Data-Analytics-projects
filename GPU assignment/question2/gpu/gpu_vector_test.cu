#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>

#define BLOCKSIZE 16

using namespace std;

float* initRandomVector(int length){
    float *vec = (float *) malloc(length * sizeof(float));
    for(int i=0;i<length;i++){
        vec[i] = rand() / (float)(RAND_MAX);
    }
    return vec;
}

__global__ void addVector(float *vecA, float *vecB, float *vecC, int length) {
    int i = (blockIdx.x * gridDim.x + blockIdx.y) * blockDim.x * blockDim.y + threadIdx.x * blockDim.x + threadIdx.y;
    if (i < length) {
        vecC[i] = vecA[i] + vecB[i];
    }
}

float sumVector(float* vec, int length){
    float sum = 0.0;
    for(int i=0;i<length;i++){
        sum += vec[i];
    }
    return sum;
}

int main(int argc, char **argv) {
    int n = atoi(argv[1]);
    srand(time(NULL));

    clock_t start, finish;
    float duration;

    /*  Initialization  */
    start = clock();

    float *host_a = initRandomVector(n);
    float *host_b = initRandomVector(n);

    finish = clock();
    duration = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Initialization: %f seconds\n", duration);

    /*  Put input onto GPU  */
    start = clock();

    cudaError_t error = cudaSuccess;
    float *device_a, *device_b, *device_c;
    error = cudaMalloc((void **)&device_a, sizeof(float) * n);
    error = cudaMalloc((void **)&device_b, sizeof(float) * n);
    error = cudaMalloc((void **)&device_c, sizeof(float) * n);

    if (error != cudaSuccess) {
        printf("Fail to cudaMalloc on GPU");
        return 1;
    }

    cudaMemcpy(device_a, host_a, sizeof(float) * n, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, host_b, sizeof(float) * n, cudaMemcpyHostToDevice);

    finish = clock();
    duration = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Put input onto GPU: %f seconds\n", duration);

    /*  Add vectors  */
    start = clock();

    int gridsize = (int)ceil(sqrt(n) / BLOCKSIZE);

    dim3 dimBlock(BLOCKSIZE, BLOCKSIZE, 1);
    dim3 dimGrid(gridsize, gridsize, 1);

    addVector<<<dimGrid, dimBlock>>>(device_a, device_b, device_c, n);
    cudaThreadSynchronize();

    finish = clock();
    duration = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Adding: %f seconds\n", duration);

    /*  Get result from gpu  */
    start = clock();
    float *host_c = (float *)malloc(sizeof(float) * n);
    cudaMemcpy(host_c, device_c, sizeof(float) * n, cudaMemcpyDeviceToHost);
    finish = clock();
    duration = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Get result from GPU: %f seconds\n", duration);

    /*  Summation  */
    start = clock();
    float sum = sumVector(host_c, n);
    finish = clock();
    duration = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Summation: %f seconds\n", duration);    

    /*  Free memory  */
    start = clock();

    free(host_a);
    free(host_b);
    free(host_c);

    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);

    finish = clock();
    duration = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Free memory: %f seconds\n", duration);

    printf("Sum = %f\n", sum);
    return 0;
}
