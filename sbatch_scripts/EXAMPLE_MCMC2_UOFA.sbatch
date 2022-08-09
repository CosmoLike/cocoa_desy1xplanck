#!/bin/bash

#SBATCH --job-name=DESY1_PL_MC2
#SBATCH --output=DESY1_PL_MC2-%j.out
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=2
#SBATCH --ntasks-per-socket=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --qos=qual_qos_timeifler
#SBATCH --account=timeifler
#SBATCH --time=24:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@email.arizona.edu

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
module load anaconda
conda init bash
source ~/.bashrc 

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd $SLURM_SUBMIT_DIR
conda activate cocoa
source start_cocoa

export OMP_PROC_BIND=close
if [ -n "$SLURM_CPUS_PER_TASK" ]; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
else
  export OMP_NUM_THREADS=1
fi

mpirun -n ${SLURM_NTASKS} --mca btl tcp,self --bind-to core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/desy1xplanck/yaml/EXAMPLE_MCMC2.yaml -r
