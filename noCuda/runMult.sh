#!/bin/bash
#$ -cwd -S /bin/bash -j y
##$ -pe smp 3
#$ -l mem=20G,time=:5:
#$ -l gpu=1
#$ -M lwz2103@c2b2.columbia.edu -m bes


# the program to run
./mult 1 1000000 400