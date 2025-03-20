#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__ void sum_array_gpu(int* a, int* b, int* c, int size)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;

    if (gid < size)
    {
        c[gid] = a[gid] + b[gid];
    }
}

void sum_arrays_cpu(int* a, int* b, int* c, int size)
{
    for (int i=0; i<size; i++)
    {
        c[i] = a[i] + b[i];
    }
}

void compare_arrays(int* a, int* b, int size)
{
    for (int i=0; i<size; i++)
    {
        if (a[i] != b[i])
        {
            printf("Arrays are different \n");
        }
    }

    printf("Arrays are same \n");
}

int main()
{
    int size = 10000;
    int block_size = 128;
    int num_bytes = size *sizeof(int);

    // host pointer arrays
    int* h_a, *h_b, *gpu_results, *h_c;

    // mem alloc host pointer arrays in heap
    h_a = (int*)malloc(num_bytes);
    h_b = (int*)malloc(num_bytes);
    h_c = (int*)malloc(num_bytes);
    gpu_results = (int*)malloc(num_bytes);

    // initialize host pointer arrays
    time_t t;
    srand( (unsigned)time(&t));
    for (int i=0; i<size; i++)
    {
        h_a[i] = (int)(rand() & 0xFF);
    }
    for (int i=0; i<size; i++)
    {
        h_b[i] = (int)(rand() & 0xFF);
    }

    // initialize device pointer arrays
    int* d_a, *d_b, *d_c;
    cudaMalloc((int**)&d_a, num_bytes);
    cudaMalloc((int**)&d_b, num_bytes);
    cudaMalloc((int**)&d_c, num_bytes);

    // memory transfer from host to device
    cudaMemcpy(d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_bytes, cudaMemcpyHostToDevice);

    //launching the grid
    dim3 block(block_size);
    dim3 grid((size/block.x)+1);

    sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);
    cudaDeviceSynchronize();

    // transfer the results from host to device
    cudaMemcpy(gpu_results, d_c, num_bytes, cudaMemcpyDeviceToHost);

    // calculate results in cpu
    sum_arrays_cpu(h_a, h_b, h_c, size);

    // compare the array result
    compare_arrays(gpu_results, h_c, size);
    // release memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(gpu_results);
    free(h_a);
    free(h_b);
    free(h_c);
}