#!/bin/bash
#$ -cwd -S /bin/bash -j y
##$ -pe smp 3
#$ -l mem=30G,time=:10:
#$ -l gpu=1
#$ -M lwz2103@c2b2.columbia.edu -m bes


# the program to run
./accuracy.o 60 1000000 400