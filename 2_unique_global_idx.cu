#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

__global__ void unique_global_thread_id(int* input)
{
    // Compute the global thread ID across all blocks
    int global_id = (blockIdx.z * gridDim.y * gridDim.x + blockIdx.y * gridDim.x + blockIdx.x) * 
                    (blockDim.z * blockDim.y * blockDim.x) + 
                    (threadIdx.z * blockDim.y * blockDim.x) + 
                    (threadIdx.y * blockDim.x) + 
                    threadIdx.x;

    printf("blockIdx.z: %d, blockIdx.y: %d, blockIdx.x: %d, threadIdx.z: %d, threadIdx.y: %d, threadIdx.x: %d, Global Thread ID: %d, Value: %d\n",
        blockIdx.z, blockIdx.y, blockIdx.x,
        threadIdx.z, threadIdx.y, threadIdx.x,
        global_id, input[global_id]);
}

int main()
{
    int grid_x = 2, grid_y = 2, grid_z = 2;   // Grid of size 2x2x1
    int block_x = 4, block_y = 4, block_z = 4; // Block of size 4x4x4

    int total_threads = (grid_x * grid_y * grid_z) * (block_x * block_y * block_z);
    int array_size = total_threads;
    int array_byte_size = sizeof(int) * array_size;
    
    printf("Total array size is: %d bytes, Total threads: %d\n", array_byte_size, total_threads);

    // Allocate host memory
    int* h_data = (int*)malloc(array_byte_size);

    // Initialize host array
    for (int i = 0; i < array_size; i++)
    {
        h_data[i] = i;
    }

    // Print host array
    for (int i = 0; i < array_size; i++)
    {
        printf("%d ", h_data[i]);
    }
    printf("\n \n");

    // Allocate device memory
    int* d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, array_byte_size);
    if (err != cudaSuccess) {
        printf("CUDA malloc failed! %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy data to device
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 grid(grid_x, grid_y, grid_z);
    dim3 block(block_x, block_y, block_z);

    // Launch kernel
    unique_global_thread_id<<<grid, block>>>(d_data);
    cudaDeviceSynchronize();

    // Free allocated memory
    cudaFree(d_data);
    free(h_data);

    cudaDeviceReset();
    return 0;
}