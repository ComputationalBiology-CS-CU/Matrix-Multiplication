NVCC = nvcc

runMatrix.o: runMatrix.cpp matrixMul.h
	$(NVCC) -c %< -o $@

matrixMul.o: matrixMul.cu
	$(NVCC) -c %< -o $@

runMatrix: runMatrix.o matrixMul.o
	$(NVCC) %^ -o $@