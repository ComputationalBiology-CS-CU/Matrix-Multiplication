CC  = gcc
CXX = g++

INCLUDES =

CFLAGS   = -g -Wall $(INCLUDES)
CXXFLAGS = -g -Wall $(INCLUDES)

LDFLAGS = -g
LDLIBS = -lstdc++

mult: mult.o

mult.o: mult.cpp

.PHONY: clean
clean:
	rm -f *.o a.out core mult

.PHONY: all
all: clean mult
