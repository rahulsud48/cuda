# Some basic cuda kernels with initial examples
#### This is not a project, it has some basic functions of cuda to understand the SIMT GPU micro-architecture
#### nvcc is required. nvcc is the compiler provided by Nvidia to run math kernels on Nvidia GPUs
#### run: nvcc <script>.cu -o <program_name>
#### nvprof was used for profiling of the kernels
#### for matrix_mult.cu --> nvcc -arch=sm_70 -o run matrix_mult.cu