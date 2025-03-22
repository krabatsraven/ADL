#!/bin/bash

#------------------------------- SBATCH -------------------------------
#SBATCH -J jobname # Job name
#SBATCH --partition=topml      # very slow: smp
#SBATCH -o outputfile.%j.out         # Specify stdout output file (%j expands to jobId)
#SBATCH --account=ki-topml
#SBATCH -t 120:00:00                  # Run time (hh:mm:ss)
#SBATCH --mem=3000
#------------------------------- Parallelize -------------------------------
#SBATCH --array=0-152:1
#------------------------------- Modules -------------------------------
module load lang/Java
#------------------------------- Virtual Environment -------------------------------
# todo: add python env here

# for CapyMOA
export MALLOC_ARENA_MAX=4
export _JAVA_OPTIONS="-Djava.io.tmpdir/localscratch/${SLURM_JOB_ID}/ -Xms50m -Xmx16g -Xss1g"

#------------------------------- Run -------------------------------
# access job array with environmental variable $SLURM_ARRAY_TASK_ID
srun python benchmark.py
#------------------------------- Deactivate Virtual Environment -------------------------------
