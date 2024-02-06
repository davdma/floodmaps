#!/bin/bash

#SBATCH --job-name=floodmodel
#SBATCH --account=HYDROSM
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=train.out
#SBATCH --error=train.err
#SBATCH --mail-user=davidma.inspire@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00

cd /lcrc/project/hydrosm/dma/src
source activate floodmapsgpu2
export PYTHONPATH=/lcrc/project/hydrosm/dma/src:$PYTHONPATH

srun python train_classifier.py --project FloodSamplesUNetRC -e 50 -b 512 -l 0.00001 --loss TverskyLoss --early_stopping
