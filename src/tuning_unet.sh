#!/bin/bash

#SBATCH --job-name=floodmodelunet
#SBATCH --account=GENERATIVE_HYDROLOGY
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/tuning.out
#SBATCH --error=outputs/tuning.err
#SBATCH --mail-user=davidma.inspire@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00

. /home/dma/miniconda3/etc/profile.d/conda.sh
cd /lcrc/project/hydrosm/dma/src
conda activate floodmaps-tuning
export PYTHONPATH=/lcrc/project/hydrosm/dma/src:$PYTHONPATH

srun python tuning_unet.py --dataset s1 -i 1 -e 10 --experiment_name sarunetcnn2
