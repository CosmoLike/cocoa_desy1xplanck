#!/bin/bash

#SBATCH --job-name=emu_train_y1
#SBATCH --output=emu_train_y1.out
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --partition=high_priority
#SBATCH --qos=user_qos_timeifler
#SBATCH --account=timeifler
#SBATCH --time=20:00:00

# Clear the environment from any previously loaded modules
module purge > /dev/null 2>&1
source ~/.bash_profile 

echo Running on host `hostname`
echo Time is `date`
echo Directory is `pwd`
echo Slurm job NAME is $SLURM_JOB_NAME
echo Slurm job ID is $SLURM_JOBID

cd /home/u7/ssarmabo/cocoa_emulator/cocoa/Cocoa/
source start_cocoa
conda activate /home/u7/ssarmabo/softs/miniconda3/envs/cocoa

CONFIGFILE=./projects/lsst_y1/yaml_files/y1_cosmic_shear/emu_test.yaml
export OMP_NUM_THREADS=1

for N in {0..3}
do
    mpirun -n 80 --mca btl tcp,self $PYTHON3 calculate_dv.py $CONFIGFILE $N
    $PYTHON3 emu_y1_cosmic_shear.py $CONFIGFILE $N 1 0
done
N=3

$PYTHON3 emu_y1_cosmic_shear.py $CONFIGFILE $N 0 1

