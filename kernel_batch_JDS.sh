#!/bin/bash
#
#SBATCH --job-name=seminarska
#SBATCH --output=outMiha.txt
#SBATCH --error=error.txt
#SBATCH --reservation=fri
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu

gcc opencl.c mtx_sparse.c -fopenmp -O2 -I/usr/include/cuda -L/usr/lib64 -l:"libOpenCL.so.1" -o out1

srun out1 data/cant.mtx SpMV_cl