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

void print_cuda_error(cudaError error)
{
    if (error != cudaSuccess)
    {
        fprintf(stderr, " Error : %s \n", cudaGetErrorString(error));
    }
}

int main()
{
    int size = 1024*1024;
    int block_size = 512;
    int num_bytes = size *sizeof(int);

    printf("The number of bytes executed : %d \n \n", num_bytes);

    // cuda error variable
    cudaError error;

    // timing
    clock_t cpu_start, cpu_end, gpu_start, gpu_end, htod_start, htod_end, dtoh_start, dtoh_end;

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
    error = cudaMalloc((int**)&d_a, num_bytes);
    print_cuda_error(error);
    error = cudaMalloc((int**)&d_b, num_bytes);
    print_cuda_error(error);
    error = cudaMalloc((int**)&d_c, num_bytes);
    print_cuda_error(error);

    // memory transfer from host to device
    htod_start = clock();
    cudaMemcpy(d_a, h_a, num_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, num_bytes, cudaMemcpyHostToDevice);
    htod_end = clock();

    //launching the grid
    dim3 block(block_size);
    dim3 grid((size/block.x)+1);

    gpu_start = clock();
    sum_array_gpu << <grid, block >> > (d_a, d_b, d_c, size);
    cudaDeviceSynchronize();
    gpu_end = clock();
    

    // transfer the results from host to device
    dtoh_start = clock();
    cudaMemcpy(gpu_results, d_c, num_bytes, cudaMemcpyDeviceToHost);
    dtoh_end = clock();

    // calculate results in cpu
    cpu_start = clock();
    sum_arrays_cpu(h_a, h_b, h_c, size);
    cpu_end = clock();

    // time taken prints
    printf("Sum array CPU execution time : %4.6f \n",(double)((double)(cpu_end - cpu_start)/CLOCKS_PER_SEC));
    printf("Total GPU Time : %4.6f \n \n",(double)((double)(dtoh_end + htod_end + gpu_end - htod_start - dtoh_start - gpu_start)/CLOCKS_PER_SEC));

    printf("Host to Device time : %4.6f \n",(double)((double)(htod_end - htod_start)/CLOCKS_PER_SEC));
    printf("Sum array GPU execution time : %4.6f \n",(double)((double)(gpu_end - gpu_start)/CLOCKS_PER_SEC));
    printf("Device to Host time : %4.6f \n",(double)((double)(dtoh_end - dtoh_start)/CLOCKS_PER_SEC));
    

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