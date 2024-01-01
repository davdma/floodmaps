#!/bin/bash

#SBATCH --job-name=floodsamples
#SBATCH --account=HYDROSM
#SBATCH --partition=bdwall
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=floodsamples.out
#SBATCH --error=floodsamples.error
#SBATCH --mail-user=davidma.inspire@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --time=60:00:00

# Set up my environment
cd /lcrc/project/hydrosm/dma
source activate floodmaps
export PYTHONPATH=/lcrc/project/hydrosm/dma/:$PYTHONPATH

# Run sampling script
srun python sample_mpc.py 200 -b 5 -a 4 -c 35 -s 40
