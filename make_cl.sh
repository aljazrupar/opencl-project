#!/bin/bash

fname=$1
gcc $fname mtx_sparse.c -fopenmp -O2 -I/usr/include/cuda -L/usr/lib64 -l:"libOpenCL.so.1" -o "${fname%%.*}"
