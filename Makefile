NVCC = nvcc
CXX = g++

runMatrix: runMatrix.o matrixMul.o
	$(NVCC) -o runMatrix runMatrix.o matrixMul.o

matrixMul.o: matrixMul.cu matrixMul.h
	$(NVCC) -c matrixMul.cu matrixmul.h

runMatrix.o: runMatrix.cpp matrixMul.h
	$(NVCC) -c runMatrix.cpp matrixMul.h



	