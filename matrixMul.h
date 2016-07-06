/*
	Reference: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.
        html#ixzz4CtH09yed 
*/

#include <cstdio>
#include <iostream>
using namespace std;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

// Thread block size
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
#define BLOCK_SIZE 20

__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);