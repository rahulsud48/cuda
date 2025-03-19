#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include<stdio.h>
#include<stdlib.h>

__global__ void unique_idx_calc_threadIdx(int* input)
{
    int tid = (gridDim.x * blockDim.x * blockIdx.y) + (blockDim.x * blockIdx.x) + threadIdx.x;
    printf("threadIdx : %d, valuse : %d \n", tid, input[tid]);
}

int main()
{
    int array_size = 16;
    int array_byte_size = sizeof(int) * array_size;
    // std::cout<<"dtype size is: "<<sizeof(int))<<"\n";
    printf("total array size is: %d \n", array_byte_size);

    int h_data[] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};

    for (int i=0; i<array_size; i++)
    {
        printf("%d ", h_data[i]);
    }

    printf("\n \n");

    int* d_data;

    cudaMalloc((void**)&d_data, array_byte_size);
    cudaMemcpy(d_data, h_data, array_byte_size, cudaMemcpyHostToDevice);

    dim3 block(4);
    dim3 grid(2,2);

    unique_idx_calc_threadIdx <<<grid, block >> > (d_data);
    cudaDeviceSynchronize();

    cudaDeviceReset();

    return 0;
}