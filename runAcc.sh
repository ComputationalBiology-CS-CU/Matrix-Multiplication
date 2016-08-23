#!/bin/bash
#$ -cwd -S /bin/bash -j y
##$ -pe smp 3
#$ -l mem=25G,time=:10:
#$ -l gpu=1
#$ -M lwz2103@c2b2.columbia.edu -m bes


# the program to run
./accuracy.o 50 1000000 400

# check for memory leaks
# cuda-memcheck ./accuracy.o a1 a2 b2
