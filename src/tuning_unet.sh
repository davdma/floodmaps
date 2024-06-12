#!/bin/bash

#SBATCH --job-name=floodmodelunet
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
conda activate floodmaps-tuning
export PYTHONPATH=/lcrc/project/hydrosm/dma/src:$PYTHONPATH

srun python tuning_unet.py --dataset s1 -i 0 -e 5 --experiment_name sarunetcnn
