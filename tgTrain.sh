#!/bin/bash

#SBATCH --verbose
#SBATCH --job-name=tgTrain
#SBATCH --time=100:00:00
#SBATCH --nodes=1
#SBATCH --mem=100GB
##SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --output=tg.train.%j

python train.py data-bin/iwslt14.tokenized.de-en --arch tg
