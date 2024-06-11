#!/bin/bash

#SBATCH --job-name=floodsamples
#SBATCH --account=HYDROSM
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=floodsamples.out
#SBATCH --error=floodsamples.error
#SBATCH --mail-user=davidma.inspire@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00

# Set up my environment
cd /lcrc/project/hydrosm/dma/sampling
source activate floodmaps-sampling
export PYTHONPATH=/lcrc/project/hydrosm/dma/:$PYTHONPATH

# Run sampling script
srun python sample_mpc.py 200 -b 5 -a 4 -c 35 -s 15 -d samples_200_5_4_35_sar/
