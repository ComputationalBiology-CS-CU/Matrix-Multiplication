/*
  Reference: http://docs.nvidia.com/cuda/cuda-c-programming-guide/index.
    html#ixzz4CtH09yed 
*/

#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <iostream>
using namespace std;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.width + col)
typedef struct {
    int width;
    int height;
    float* elements;
} Matrix;

// Thread block size
#define BLOCK_SIZE 8

// Forward declaration of the matrix multiplication kernel
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);

// Matrix multiplication - Host code
// Matrix dimensions are assumed to be multiples of BLOCK_SIZE
void MatMul(const Matrix A, const Matrix B, Matrix C)
{
    // Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; 
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    cudaError_t err = cudaMalloc(&d_A.elements, size);
    cout << "CUDA malloc A: " << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(d_A.elements, A.elements, size, cudaMemcpyHostToDevice);
    cout << "Copy A to device: " << cudaGetErrorString(err) << "\n" << endl;

    Matrix d_B;
    d_B.width = B.width; 
    d_B.height = B.height;
    size = B.width * B.height * sizeof(float);
    err = cudaMalloc(&d_B.elements, size);
    cout << "CUDA malloc B: " << cudaGetErrorString(err) << endl;
    err = cudaMemcpy(d_B.elements, B.elements, size, cudaMemcpyHostToDevice);
    cout << "Copy B to device: " << cudaGetErrorString(err) << "\n" << endl;

    // Allocate C in device memory
    Matrix d_C;
    d_C.width = C.width; 
    d_C.height = C.height;
    size = C.width * C.height * sizeof(float);
    err = cudaMalloc(&d_C.elements, size);
    cout << "CUDA malloc C: " << cudaGetErrorString(err) << endl;

    // Invoke kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((B.width + dimBlock.x - 1) / dimBlock.x, 
        (A.height + dimBlock.y - 1) / dimBlock.y);
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
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    for (int e = 0; e < A.width; ++e)
        Cvalue += A.elements[row * A.width + e]
                * B.elements[e * B.width + col];
    C.elements[row * C.width + col] = Cvalue;
}

int main(int argc, char const *argv[])
{
    clock_t t;
    Matrix A, B, C;
    int a1, a2, b1, b2;
    int i, j;

    srand(time(NULL));

    // Get dimensions of A and B
    // Run $ ./matrixMul 1 1000000 400
    a1 = atoi(argv[1]); // A's height
    a2 = atoi(argv[2]); // A's width
    b1 = a2; // B's height
    b2 = atoi(argv[3]); // B's width

    A.height = a1;
    A.width = a2;
    A.elements = new float[A.width * A.height];

    B.height = b1;
    B.width = b2;
    B.elements = new float[B.width * B. height];

    C.height = A.height;
    C.width = B.width;
    C.elements = new float[C.width * C.height];

    // Fill A and B with random floats
    for (i = 0; i < A.height; ++i)
        for (j = 0; j < A.width; ++j)
            A.elements[i * A.width + j] = float(rand() % 100);

    for (i = 0; i < B.height; ++i)
        for (j = 0; j < B.width; ++j)
            B.elements[i * B.width + j] = float(rand() % 100);

    // Call MatMul(), and therefore MatMulKernel()
    t = clock();

    MatMul(A, B, C);

    // Print time multiplication took
    t = clock() - t;
    cout << "It took me " << fixed << ((float)t)/CLOCKS_PER_SEC;
    cout << " seconds.\n" << endl;

    // Print A, B, and C
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

    for (i = 0; i < min(10, C.height); ++i) {
        for (j = 0; j < min(10, C.width); ++j)
            cout << fixed << C.elements[i * C.width + j] << "\t";

        cout << endl;
    }
    cout << endl;

    delete[] A.elements;
    delete[] B.elements;
    delete[] C.elements;
    
    return 0;
}