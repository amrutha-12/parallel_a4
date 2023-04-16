#include<stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#define max 250

void init_matrix(int n, int32_t *matrix)
{
    for(int i=0; i<n; i++)
    {
        // matrix[i] = (int32_t*)malloc(n * sizeof(int32_t);
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
            printf("%d ",mat[i*n+j]);
        }
        printf("/n");
    }
}

__global__ void matmul(int32_t *A, int32_t *B, int32_t *C, int n)
{   
    int ind = (blockIdx.x * 1024) + threadIdx.x;
    int i = blockIdx.x/(n/1024);
    int j = threadIdx.x + (blockIdx.x%(n/1024))*1024;
    if (i<n && j<n)
    {
        for(int k=0;k<n;k++)
            {
                C[ind] += (A[i*n+k]*B[k*n+j]);
            }
    }
}

// int validate(int32_t *A, int32_t *B, int32_t *C, int n)
// {
//     int errors = 0;
//     for (int i=0;i<n;i++)
//     {
//         for (int j=0;j<n;j++)
//         {
//             int32_t C_check = 0;
//             for (int k=0;k<n;k++)
//             {
//                 C_check+=A[i*n+k]*B[k*n+j];
//             }
//             if(C_check!=C[i*n+j])
//             {
//                 errors+=1;
//             }
//         }
//     }
//     return errors;
// }

void validate(int32_t *A,int32_t *B,int32_t *C,int n)
{
    int mistakes=0;
    //Serial Matmul
    for(int i=0;i<n;i++)
    {
        for(int j=0;j<n;j++)
        {
            int32_t temp = 0;
            for(int k=0;k<n;k++)
            {
                temp+=A[i*n+k]*B[k*n+j];
            }
            if(temp != C[i*n+j])
                mistakes++;
        }
    }
    printf("#Mistakes: %d\n",mistakes);
}

int main(int argc, char**argv)
{
    int n = atoi(argv[1]);
    int32_t* A = (int32_t*)malloc(n * n * sizeof(int32_t));
    int32_t* B = (int32_t*)malloc(n * n * sizeof(int32_t));
    int32_t* C = (int32_t*)malloc(n * n * sizeof(int32_t));
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
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //Allocating memory space on the device
    int32_t *d_A, *d_B, *d_C;
    cudaMalloc((void **) &d_A, sizeof(int32_t)*n*n);
    cudaMalloc((void **) &d_B, sizeof(int32_t)*n*n);
    cudaMalloc((void **) &d_C, sizeof(int32_t)*n*n);

    // Recording time taken to transfer data from host to device
    cudaEventRecord(start,0);

    //Copying matrices A,B,C from host to device memory
    cudaMemcpy(d_A, A, sizeof(int32_t)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeof(int32_t)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, C, sizeof(int32_t)*n*n, cudaMemcpyHostToDevice);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&data_transfer_time1, start, stop);

    // Recording time taken to compute matrix multiplication
    cudaEventRecord(start,0);
    matmul<<<n*(n/1024),1024>>>(d_A,d_B,d_C,n);

    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&computation_time, start, stop);

    // Recording time taken to transfer result from device to host memory
    cudaEventRecord(start,0);
    cudaMemcpy(C,d_C, sizeof(int32_t)*n*n, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop,0);
    cudaEventElapsedTime(&data_transfer_time2, start, stop);

    // int mistakes = validate(A,B,C,n);
    // if(mistakes!=0)
    // {
    //     printf("Matrix multiplication was not correct, %d mistakes were reported\n", mistakes);
    // }
    // else
    // {
    //     printf("Matrix multiplication was successfull!\n");
    // }

    if(true)
    {
        validate(A,B,C,n);
    }

    printf("Time taken to transfer data: %f\n", data_transfer_time1+data_transfer_time2);
    printf("Time taken to perform the computation: %f\n",computation_time);

    free(A);
    free(B);
    free(C);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);

    return 0;
}