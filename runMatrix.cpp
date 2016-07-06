#import "matrixMul.h"

int main(int argc, char const *argv[])
{
	Matrix A, B, C;
	int a1, a2, b1, b2;

	// Get dimensions of A and B
	// Run $ ./runMatrix 1 1000000 400
	a1 = atoi(argv[1]); // A's height
	a2 = atoi(argv[2]); // A's width
	b1 = a2; // B's height
	b2 = atoi(argv[3]) // B's width

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
			A.elements[i * A.width + j] = (arc4random() % 3);

	for (int i = 0; i < B.height; ++i)
		for (int j = 0; j < B.width; ++j)
			B.elements[i * B.width + j] = (arc4random() % 2);

	// Call MatMul(), and therefore MatMulKernel()
	MatMul(A, B, C);

	// Print A, B, and C
	for (int i = 0; i < min(10, A.height); ++i) {
		for (int j = 0; i < min(10, A.width); ++j)
			cout << fixed << A.elements[i * A.width + j];

		cout << endl;
	}
	cout << endl;

	for (int i = 0; i < min(10, B.height); ++i) {
		for (int j = 0; i < min(10, B.width); ++j)
			cout << fixed << B.elements[i * B.width + j];

		cout << endl;
	}
	cout << endl;

	for (int i = 0; i < min(10, C.height); ++i) {
		for (int j = 0; i < min(10, C.width); ++j)
			cout << fixed << C.elements[i * C.width + j];

		cout << endl;
	}
	cout << endl;

	// Do you need these??
	delete [] A.elements;
	delete [] B.elements;
	delete [] C.elements;
	
	return 0;
}