#include <stdio.h>
#include <stdlib.h>
#include <ctime>

using namespace std;

float* initRandomVecotr(int length){
    float *vec = (float *) malloc(length * sizeof(float));
    for(int i=0;i<length;i++){
        vec[i] = rand() / (float)(RAND_MAX);
    }
    return vec;
}

float* addVector(float *vecA, float *vecB, int length){
    float *vecC = (float *) malloc(length * sizeof(float));
    if (!vecC){
        printf("Out of memory\n");
        exit(-1);
    }
    for(int i=0;i<length;i++){
        vecC[i] = vecA[i] + vecB[i];
    }
    return vecC;
}

float sumVector(float* vec, int length){
    float sum = 0.0;
    for(int i=0;i<length;i++){
        sum += vec[i];
    }
    return sum;
}

int main(int argc, char *argv[]){
    int length = atoi(argv[1]);
    srand(time(NULL));

    clock_t start_total, finish_total;
    clock_t start, finish;
    float duration;

    start_total = clock(); 

    start = clock(); 
    float *vecA = initRandomVecotr(length);
    float *vecB = initRandomVecotr(length);
    finish = clock();
    duration = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Initialization: %f seconds\n", duration);

    start = clock();
    float *vecC = addVector(vecA, vecB, length);
    finish = clock();
    duration = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Adding: %f seconds\n", duration);

    start = clock();
    float sum = sumVector(vecC, length);
    finish = clock();
    duration = (float)(finish - start) / CLOCKS_PER_SEC;
    printf("Summation: %f seconds\n", duration);

    printf("Sum = %f\n", sum);

    finish_total = clock();
    duration = (float)(finish_total - start_total) / CLOCKS_PER_SEC;
    printf("Total time:\t%f seconds\n", duration);
    
    return 0;
}