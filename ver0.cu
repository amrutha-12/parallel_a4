#include<stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <iostream>

using namespace std;

#define max 250

void init_matrix(int n, int32_t *matrix)
{
    for(int i=0; i<n; i++)
    {
        for(int j=0; j<n; j++)
        {
            matrix[(i*n)+j]=(rand()%max);
        }
    }
}

void printMat(int32_t *mat, int n)
{
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            cout<<mat[i*n+j]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}

__global__ void matmul(int32_t *A, int32_t *B, int32_t *C, int n)
{
    int index = (blockIdx.x * min(n,1024)) + threadIdx.x;
    int i = blockIdx.x/(n/min(n,1024));
    int j = threadIdx.x + (blockIdx.x%(n/min(n,1024)))*min(n,1024);
    int32_t sum = 0;
    for(int k=0;k<n;k++)
    {
        sum += A[i*n + k]*B[k*n+j];
    }
    C[index] = sum;
}

int validate(int32_t *A, int32_t *B, int32_t *C, int n)
{
    int errors = 0;
    for (int i=0;i<n;i++)
    {
        for (int j=0;j<n;j++)
        {
            int32_t C_check = 0;
            for (int k=0;k<n;k++)
            {
                C_check+=A[i*n+k]*B[k*n+j];
            }
            if(C_check!=C[i*n+j])
            {
                errors+=1;
            }
        }
    }
    return errors;
}

int main(int argc, char**argv)
{
    cudaError_t err;
    if(argc < 2) {
        cout << "usage: ./a.out <size_of_matrix>";
        return 0;
    }
    int n = atoi(argv[1]);
    int32_t *A = (int32_t*)malloc(n * n * sizeof(int32_t));
    int32_t *B = (int32_t*)malloc(n * n * sizeof(int32_t));
    int32_t *C = (int32_t*)malloc(n * n * sizeof(int32_t));
    for(int i=0;i<n;i++)
    {
        for (int j=0;j<n;j++)
        {
            C[i*n+j]=0;
        }
    }
    init_matrix(n,A);
    init_matrix(n,B);

    float data_transfer_time1, data_transfer_time2, computation_time;
    // some events to calculate execution time
    cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaEventCreate(&stop);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }

    //Allocating memory space on the device
    int32_t *d_A, *d_B, *d_C;
    err = cudaMalloc((void**)&d_A, n*n*sizeof(int32_t));
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaMalloc((void**)&d_B, n*n*sizeof(int32_t));
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaMalloc((void**)&d_C, n*n*sizeof(int32_t));
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }

    // Recording time taken to transfer data from host to device
    err = cudaEventRecord(start,0);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }

    //Copying matrices A,B,C from host to device memory
    err = cudaMemcpy(d_A, A, n*n*sizeof(int32_t), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaMemcpy(d_B, B, n*n*sizeof(int32_t), cudaMemcpyHostToDevice);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    // cudaMemcpy(d_C, C, sizeof(int32_t)*n*n, cudaMemcpyHostToDevice);

    err = cudaEventRecord(stop, 0);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaEventSynchronize(stop);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaEventElapsedTime(&data_transfer_time1, start, stop);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    // Recording time taken to compute matrix multiplication
    err = cudaEventRecord(start,0);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    dim3 threadsPerBlock(min(n, 1024));
    dim3 numBlocks(n*n/min(n, 1024));
    matmul<<<numBlocks,threadsPerBlock>>>(d_A,d_B,d_C,n);
    err = cudaEventRecord(stop,0);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaEventSynchronize(stop);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaEventElapsedTime(&computation_time, start, stop);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }

    // Recording time taken to transfer result from device to host memory
    err = cudaEventRecord(start,0);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaMemcpy(C,d_C, n*n*sizeof(int32_t), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaEventRecord(stop,0);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaEventElapsedTime(&data_transfer_time2, start, stop);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }

    int mistakes = validate(A,B,C,n);
    if(mistakes!=0)
    {
        printf("Matrix multiplication was not correct, %d mistakes were reported\n", mistakes);
    }
    else
    {
        printf("Matrix multiplication was successfull!\n");
    }

    printf("Time taken to transfer data: %f\n", data_transfer_time1+data_transfer_time2);
    printf("Time taken to perform the computation: %f\n",computation_time);
    free(A);
    free(B);
    free(C);
    err = cudaFree(d_A);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaFree(d_B);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }
    err = cudaFree(d_C);
    if(err != cudaSuccess) {
        cout<<cudaGetErrorString(err)<<endl;
        exit(1);
    }

    return 0;
}