#!/bin/bash

#SBATCH --job-name=floodmodel
#SBATCH --account=HYDROSM
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=train_sar.out
#SBATCH --error=train_sar.err
#SBATCH --mail-user=davidma.inspire@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --time=6:00:00

. /home/dma/miniconda3/etc/profile.d/conda.sh
cd /lcrc/project/hydrosm/dma/src
conda activate floodmaps-training
export PYTHONPATH=/lcrc/project/hydrosm/dma/src:$PYTHONPATH

srun python train_sar.py --project dummytest --name 'unet++' --autodespeckler 'CNN' -e 1 -l 0.0007325 -b 256 -c 1111111 --loss BCELoss --early_stopping --num_workers 10 --dropout 0.1081 --patience 10 --subset 0.05
