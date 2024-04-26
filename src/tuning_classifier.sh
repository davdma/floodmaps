#!/bin/bash

#SBATCH --job-name=floodmodel
#SBATCH --account=HYDROSM
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=tuning.out
#SBATCH --error=tuning.err
#SBATCH --mail-user=davidma.inspire@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00

. /home/dma/miniconda3/etc/profile.d/conda.sh
cd /lcrc/project/hydrosm/dma/src
conda activate floodmapsdp
export PYTHONPATH=/lcrc/project/hydrosm/dma/src:$PYTHONPATH

srun python tuning_classifier.py -i 0 -e 1
