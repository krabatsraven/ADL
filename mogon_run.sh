#!/bin/bash

#------------------------------- SBATCH -------------------------------
#SBATCH -J bench_adl # Job name
#SBATCH --partition=parallel      # very slow: smp
#SBATCH -o outputfile.%j.out         # Specify stdout output file (%j expands to jobId)
#SBATCH --account=m2_datamining
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss)
#SBATCH --mem=3000
#------------------------------- Parallelize -------------------------------
#SBATCH --array=375-380:1
#------------------------------- Modules -------------------------------
module load lang/Java
#------------------------------- Virtual Environment -------------------------------
source /home/djacoby/.bashrc
conda_initialize
micromamba activate adl_env

# for CapyMOA
export MALLOC_ARENA_MAX=4
export _JAVA_OPTIONS="-Djava.io.tmpdir/localscratch/${SLURM_JOB_ID}/ -Xms50m -Xmx16g -Xss1g"

#------------------------------- Run -------------------------------
# access job array with environmental variable $SLURM_ARRAY_TASK_ID
srun python benchmark.py
#------------------------------- Deactivate Virtual Environment -------------------------------
