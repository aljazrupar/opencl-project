#!/bin/bash
#
#SBATCH --job-name=seminarska
#SBATCH --output=out.txt
#SBATCH --error=error.txt
#SBATCH --reservation=fri
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --constraint=gpu


srun ./out1 test.mtx SpMV_cl