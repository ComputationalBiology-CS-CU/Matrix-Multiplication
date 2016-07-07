NVCC = nvcc
CXX = g++

runMatrix: runMatrix.o matrixMul.o
	$(NVCC) %^ -o $@

runMatrix.o: runMatrix.cpp matrixMul.h
	$(NVCC) -c %< -o $@

matrixMul.o: matrixMul.cu matrixMul.h
	$(NVCC) -c %< -o $@

	