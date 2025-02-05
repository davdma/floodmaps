#!/bin/bash

#SBATCH --job-name=floodmodel
#SBATCH --account=HYDROSM
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/train_sar.out
#SBATCH --error=outputs/train_sar.err
#SBATCH --mail-user=davidma.inspire@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00

. /home/dma/miniconda3/etc/profile.d/conda.sh
cd /lcrc/project/hydrosm/dma/src
conda activate floodmaps-training
export PYTHONPATH=/lcrc/project/hydrosm/dma/src:$PYTHONPATH

srun python train_sar.py --project dummytest --name 'unet++' --deep_supervision --autodespeckler 'VAE' -e 150 -l 0.0007325 -b 256 -c 1111111 --loss BCELoss --early_stopping --num_workers 10 --dropout 0.1081 --patience 10 --subset 1.0 --latent_dim 128 --VAE_beta 5 --shift_invariant
