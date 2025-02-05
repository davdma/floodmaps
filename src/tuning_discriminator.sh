#!/bin/bash

#SBATCH --job-name=floodmodeldisc
#SBATCH --account=HYDROSM
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --output=outputs/tuningdisc2.out
#SBATCH --error=outputs/tuningdisc2.err
#SBATCH --mail-user=davidma.inspire@outlook.com
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00

. /home/dma/miniconda3/etc/profile.d/conda.sh
cd /lcrc/project/hydrosm/dma/src
conda activate floodmaps-tuning
export PYTHONPATH=/lcrc/project/hydrosm/dma/src:$PYTHONPATH

srun python tuning_discriminator.py -i 1 -e 18 --early_stopping
