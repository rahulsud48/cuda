#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>


int main()
{
    
    // memory size 32 x 4(FP32 = 4 Bytes) = 128 MB
    int isize = 1<<25;
    int nbytes = isize * sizeof(float);

    // allocate the host memory
    float *h_a = (float*)malloc(nbytes);
    //cudaMallocHost((float **)&h_a, nbytes);

    // allocate the device memory
    float *d_a;
    cudaMalloc((float **)&d_a, nbytes);

    // initialize the host memory
    for(int i=0; i<isize; i++)
    {
        h_a[i] = 7;

    }

    // data transfer from host to device
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyHostToDevice);

    // data transfer from device to host
    cudaMemcpy(d_a, h_a, nbytes, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_a);
    free(h_a);
    // cudaFreeHost(h_a);


    return 0;
}