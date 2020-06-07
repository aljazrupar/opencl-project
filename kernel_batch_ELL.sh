#!/bin/bash
#
#SBATCH --job-name=seminarska
#SBATCH --output=aa_outELL.txt
#SBATCH --error=error.txt
#SBATCH --reservation=fri
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu

gcc opencl_ELL.c mtx_sparse.c -fopenmp -O2 -I/usr/include/cuda -L/usr/lib64 -l:"libOpenCL.so.1" -o outELL

srun outELL newData/cant_copied.mtx SpMV_cl