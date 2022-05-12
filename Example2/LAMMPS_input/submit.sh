#!/bin/csh
#$ -N CH4_TraPPE
#$ -t 1-16:1
#$ -r y
#$ -pe mpi-16 16 
#$ -q long


module load lammps
mpiexec -n $NSLOTS lmp_linux < ${SGE_TASK_ID}.TraPPE 
