#!/bin/bash

#SBATCH --job-name=floodmodelunetpp
#SBATCH --account=GENERATIVE_HYDROLOGY
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/tuning_unet_plus.out
#SBATCH --error=outputs/tuning_unet_plus.err
#SBATCH --mail-user=davidma.inspire@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00

. /home/dma/miniconda3/etc/profile.d/conda.sh
cd /lcrc/project/hydrosm/dma/src
conda activate floodmaps-tuning
export PYTHONPATH=/lcrc/project/hydrosm/dma/src:$PYTHONPATH

srun python tuning_unet_plus.py --dataset s2 -i 1 -e 30 --experiment_name plusdrops2
