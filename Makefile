NVCC = nvcc

runMult: runMult.o multShared.o

runMult.o: runMult.cpp mult.h

multShared.o: multShared.cu mult.h