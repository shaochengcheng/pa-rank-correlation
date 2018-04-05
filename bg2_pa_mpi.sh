#!/bin/bash
#PBS -l nodes=4:ppn=32
#PBS -l walltime=00:30:00
#PBS -N pa-rank-correlation
#PBS -q cpu
#PBS -V
#PBS -l gres=ccm

module swap PrgEnv-cray PrgEnv-gnu
module load ccm
module load openmpi/ccm/gnu/1.8.4
source activate gtpy3

ccmrun mpiexec -np 128 python pa_mpi.py

