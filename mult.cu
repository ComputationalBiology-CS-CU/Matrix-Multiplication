#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <iostream>
#include <cmath>
#include <cublas_v2.h>
using namespace std;

// Matrices are stored in row-major order:
// M(row, col) = *(M.elements + row * M.stride + col)
typedef struct {
    int width;
    int height;
    int stride; 
    float* elements;
} Matrix;

void MatMul(const Matrix A, const Matrix B, Matrix C)
{
	cudaError_t err;
	cublasStatus_t stat;
	cublasHandle_t handle;

	// Create cuBLAS handle
	stat = cublasCreate(&handle);
	if (stat != CUBLAS_STATUS_SUCCESS)
		cout << "CUBLAS initialization failed\n" << endl;

	// Load A and B to device memory
    Matrix d_A;
    d_A.width = A.width; 
    d_A.height = A.height;
    size_t size = A.width * A.height * sizeof(float);
    err = cudaMalloc(&d_A.elements, size);
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

    // Matrix multiplication
    /*
    	cuBLAS stores column-major matrices, C/C++ row-major
    	A and B don't need to be transposed, flipping their order is sufficient
    	ex) B * A rather than A * B
    */
    float alpha = 1.0f, beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
             B.width, A.height, A.width, &alpha,
             B.elements, B.width, A.elements, A.width,
             &beta, C.elements, C.width);

    // Read C from device memory
    err = cudaMemcpy(C.elements, d_C.elements, size, cudaMemcpyDeviceToHost);
    cout << "Copy C off of device: " << cudaGetErrorString(err) << "\n" << endl;

    // Free device memory
    cudaFree(d_A.elements);
    cudaFree(d_B.elements);
    cudaFree(d_C.elements);

    // Destroy cuBLAS handle
    cublasDestroy(handle);
}

int main(int argc, char const *argv[])
{
    clock_t t;
    Matrix A, B, C;
    int a1, a2, b1, b2;
    int i, j;

    srand(time(NULL));

    // Print usage if not provided with matrix dimensions
    if (argc < 4)
        cout << "Usage: ./accuracy.o A.height A.width B.width" << endl;

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

    // Fill A and B with random floats
    for (i = 0; i < A.height; ++i) 
        for (j = 0; j < A.width; ++j) 
            A.elements[i * A.width + j] = ((float)rand() / (float)RAND_MAX) * 100;

    for (i = 0; i < B.height; ++i) 
        for (j = 0; j < B.width; ++j) 
            B.elements[i * B.width + j] = ((float)rand() / (float)RAND_MAX) * 100;

    // Call MatMul() to perform multiplication
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