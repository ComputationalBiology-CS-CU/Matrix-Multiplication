/*
    Reference: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.
        html#ixzz4CtH09yed 
*/

#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <cmath>
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
#define BLOCK_SIZE 2

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Get a matrix element
__device__ float GetElement(const Matrix A, int row, int col)
{
    return A.elements[row * A.stride + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col,
                           float value)
{
    A.elements[row * A.stride + col] = value;
}

// Get the BLOCK_SIZExBLOCK_SIZE sub-matrix Asub of A that is
// located col sub-matrices to the right and row sub-matrices down
// from the upper-left corner of A
 __device__ Matrix GetSubMatrix(Matrix A, int row, int col) 
{
    Matrix Asub;
    Asub.width    = BLOCK_SIZE;
    Asub.height   = BLOCK_SIZE;
    Asub.stride   = A.stride;
    Asub.elements = &A.elements[A.stride * BLOCK_SIZE * row
                                         + BLOCK_SIZE * col];

    return Asub;
}

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = d_A.stride = A.width; 
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaError_t err = cudaMalloc(&d_A.elements, size);
    cout << "CUDA malloc A: " << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    cout << "Copy A to device: " << cudaGetErrorString(err) << "\n" << endl;

    Matrix d_B;
    d_B.width = d_B.stride = B.width; 
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    err = cudaMalloc(&d_B.elements, size);
    cout << "CUDA malloc B: " << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    cout << "Copy B to device: " << cudaGetErrorString(err) << "\n" << endl;
    
    // Allocate C in device memory
    Matrix d_C;
    d_C.width = d_C.stride = C.width; 
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    err = cudaMalloc(&d_C.elements, size);
    cout << "CUDA malloc C: " << cudaGetErrorString(err) << endl;

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(B.width / dimBlock.x, A.height / dimBlock.y);
    MatMulKernel<<<dimGrid, dimBlock>>>(d_A, d_B, d_C);
    err = cudaThreadSynchronize();
    cout << "Run kernel: " << cudaGetErrorString(err) << endl;

    // Read C from device memory
    err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cout << "Copy C off of device: " << cudaGetErrorString(err) << "\n" << endl;

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);
}

// Matrix multiplication kernel called by MatMul()
 __global__ void MatMulKernel(Matrix A, Matrix B, Matrix C)
{
    
    // Shared memory used to store Asub and Bsub respectively
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0.0;

    for (int i = 0; i < (A.width - 1)/BLOCK_SIZE + 1; ++i) {
        int temp = i * BLOCK_SIZE + threadIdx.x;
        if (row < A.height && temp < A.width)
            As[threadIdx.y][threadIdx.x] = A.elements[row * A.width + temp];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        temp = i * BLOCK_SIZE + threadIdx.y;
        if (col < B.width && temp < B.height)
            Bs[threadIdx.y][threadIdx.x] = B.elements[temp * B.width + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0;

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; ++j)
            Cvalue += As[threadIdx.y][j] * Bs[j][threadIdx.x];

        __syncthreads();
    }

    if (row < C.height && col < C.width)
        C.elements[row * C.width + col] = Cvalue;
    
    /*---Original code from CUDA C Programming Guide---*/
    /*
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // Each thread block computes one sub-matrix Csub of C
    Matrix Csub = GetSubMatrix(C, blockRow, blockCol);

    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;

    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;

    // Loop over all the sub-matrices of A and B that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (A.width / BLOCK_SIZE); ++m) {

        // Get sub-matrix Asub of A
        Matrix Asub = GetSubMatrix(A, blockRow, m);

        // Get sub-matrix Bsub of B
        Matrix Bsub = GetSubMatrix(B, m, blockCol);

        // Shared memory used to store Asub and Bsub respectively
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        As[row][col] = GetElement(Asub, row, col);
        Bs[row][col] = GetElement(Bsub, row, col);

        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += As[row][e] * Bs[e][col];

        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }

    // Write Csub to device memory
    // Each thread writes one element
    SetElement(Csub, row, col, Cvalue);
    */
}

int main(int argc, char const *argv[])
{
    Matrix A, B, C, D;
    int a1, a2, b1, b2;
    int i, j, k;
    float sum = 0.0, param = 0.0, square = 0.0;

    srand(time(NULL));

    // Get dimensions of A and B
    // Run $ ./matrixMul 1 1000000 400
    a1 = atoi(argv[1]); // A's height
    a2 = atoi(argv[2]); // A's width
    b1 = a2; // B's height
    b2 = atoi(argv[3]); // B's width

    A.height = a1;
    A.width = A.stride = a2;
    A.elements = new float[A.width * A.height];

    B.height = b1;
    B.width = B.stride = b2;
    B.elements = new float[B.width * B. height];

    C.height = A.height;
    C.width = C.stride = B.width;
    C.elements = new float[C.width * C.height];

    D.height = A.height;
    D.width = D.stride = B.width;
    D.elements = new float[D.width * D.height];

    // Fill A and B with random floats
    for (i = 0; i < A.height; ++i) 
        for (j = 0; j < A.width; ++j) 
            A.elements[i * A.width + j] = ((float)rand() / (float)RAND_MAX) * 100;

    for (i = 0; i < B.height; ++i) 
        for (j = 0; j < B.width; ++j) 
            B.elements[i * B.width + j] = ((float)rand() / (float)RAND_MAX) * 100;
    /*
    for (i = 0; i < D.height; ++i) 
        for (j = 0; j < D.width; ++j) 
            D.elements[i * D.width + j] = ((float)rand() / (float)RAND_MAX) * 100;
    */

    // Vanilla C++ matrix multiplication
    for (i = 0; i < A.height; ++i)
        for (j = 0; j < B.width; ++j)
            for (k = 0; k < A.width; ++k) {
                C.elements[i * C.width + j] += A.elements[i * A.width + k] * 
                    B.elements[k * B.width + j];
            }

    // Call MatMul(), and therefore MatMulKernel()
    MatMul(A, B, D);

    // Compare matrices C and D -- they should be almost identical
    cout << "Accuracy is: ";

    for (i = 0; i < C.height; ++i) {
        for (j = 0; j < C.width; ++j) {
            param = C.elements[i * C.width + j] - D.elements[i * D.width + j];            
            if (param < 0)
                param = fabsf(param);
            /*--TEST--*/
            float test = 5.0 - 6.0;
            cout << "absolute value of -1 is " << fabsf(test) << endl;

            cout << "param is " << param << endl;

            square = pow(param, 2);
            cout << "square is " << square << endl;

            sum += square;
            cout << "sum is " << sum << endl;
        }
    }
    cout << fixed << sum << endl;

    // Print matrices A, B, C, and D
    for (i = 0; i < min(10, A.height); ++i) {
        for (j = 0; j < min(10, A.width); ++j)
            cout << fixed << A.elements[i * A.width + j] << "\t";
        
        cout << endl;
    }
    cout << endl;
    
    for (i = 0; i < min(10, B.height); ++i) {
        for (j = 0; j < min(10, B.width); ++j)
            cout << fixed << B.elements[i * B.width + j] << "\t";

        cout << endl;
    }
    cout << endl;

    for (int i = 0; i < min(10, C.height); ++i) {
        for (int j = 0; j < min(10, C.width); ++j) {
            cout << fixed << C.elements[i * C.width + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;

    for (int i = 0; i < min(10, D.height); ++i) {
        for (int j = 0; j < min(10, D.width); ++j) {
            cout << fixed << D.elements[i * D.width + j] << "\t";
        }
        cout << endl;
    }
    cout << endl;
    
    delete[] A.elements;
    delete[] B.elements;
    delete[] C.elements;
    delete[] D.elements;
    
    return 0;
}