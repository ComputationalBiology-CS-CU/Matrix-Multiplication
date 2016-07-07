#include <cstdlib>
#include <ctime>
#include "matrixMul.h"

int main(int argc, char const *argv[])
{
    clock_t t;
    Matrix A, B, C;
    int a1, a2, b1, b2;

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
    for (int i = 0; i < A.height; ++i)
        for (int j = 0; j < A.width; ++j)
            A.elements[i * A.width + j] = float(rand() % 100);

    for (int i = 0; i < B.height; ++i)
        for (int j = 0; j < B.width; ++j)
            B.elements[i * B.width + j] = float(rand() % 100);

    // Call MatMul(), and therefore MatMulKernel()
    t = clock();

    MatMul(A, B, C);

    // Print time the multiplication took
	t = clock() - t;
    cout << "It took me " << fixed << ((float)t)/CLOCKS_PER_SEC;
    cout << " seconds." << endl;

    // Print C
    for (int i = 0; i < min(10, C.height); ++i) {
        for (int j = 0; j < min(10, C.width); ++j)
            cout << fixed << C.elements[i * C.width + j] << "\t";

        cout << endl;
    }
    cout << endl;

    delete [] A.elements;
    delete [] B.elements;
    delete [] C.elements;
    
    return 0;
}