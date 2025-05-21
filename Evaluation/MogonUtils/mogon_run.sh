#!/bin/bash
# job script specific for mogon 2 add partition and account before running, and username

#------------------------------- SBATCH -------------------------------
#SBATCH -J bench_adl # Job name
#SBATCH --partition=TO ADD
#SBATCH -o outputfile.%j.out         # Specify stdout output file (%j expands to jobId)
#SBATCH --account=TO ADD
#SBATCH -t 02:00:00                  # Run time (hh:mm:ss)
#SBATCH --mem=3000
#------------------------------- Parallelize -------------------------------
#SBATCH --array=0-380:1
#------------------------------- Modules -------------------------------
module load lang/Java
#------------------------------- Virtual Environment -------------------------------
source /home/USERNAME/.bashrc
conda_initialize
micromamba activate adl_env

# for CapyMOA
export MALLOC_ARENA_MAX=4
export _JAVA_OPTIONS="-Djava.io.tmpdir/localscratch/${SLURM_JOB_ID}/ -Xms50m -Xmx16g -Xss1g"

#------------------------------- Run -------------------------------
# access job array with environmental variable $SLURM_ARRAY_TASK_ID
srun python benchmark.py
#------------------------------- Deactivate Virtual Environment -------------------------------
