#!/bin/bash

#SBATCH --job-name=MCMC_DY1xP
#SBATCH --output=eval-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --ntasks-per-socket=4
#SBATCH --cpus-per-task=8
#SBATCH --partition=standard
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler
#SBATCH --time=100:00:00
#SBATCH --export=none
#SBATCH --mail-type=ALL
#SBATCH --mail-user=jiachuanxu@email.arizona.edu

#module load gsl/2.6
module load anaconda

source ~/.bashrc
echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd $SLURM_SUBMIT_DIR
#conda init bash
conda activate cocoa
source start_cocoa
#source ./projects/desy1xplanck/scripts/start_desy1xplanck

export OMP_PROC_BIND=close
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo Number of tasks is ${SLURM_NTASKS}
echo OMP threads is ${OMP_NUM_THREADS}

### Close CPU binding request
mpirun -n ${SLURM_NTASKS} --mca btl tcp,self --bind-to core --rank-by core --map-by numa:pe=${OMP_NUM_THREADS} cobaya-run ./projects/desy1xplanck/EXAMPLE_MCMC2.yaml -r

### Arbitrary CPU allocation
#mpirun -n ${SLURM_NTASKS} --mca btl tcp,self cobaya-run ./projects/desy1xplanck/EXAMPLE_MCMC2.yaml -r

#source ./projects/desy1xplanck/scripts/stop_desy1xplanck
#source stop_cocoa
#conda deactivate
