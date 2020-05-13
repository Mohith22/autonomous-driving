#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --mem=50GB
#SBATCH --job-name=GPUDemo
#SBATCH --mail-type=END
#SBATCH --mail-user=aar653@nyu.edu
#SBATCH --output=slurm_%j.out

source activate condaenv3
#python3 main_depth.py --data_dir "../../../../data" --depth_dir "../../../../data_dir" --annotation_dir "../../../../data/annotation.csv" --model_dir "Siamese_orient_depth_sbatch_dice_v100" --cuda --num_train_epochs 50 --per_gpu_batch_size 2 --depth_avail True --loss dice --siamese True --use_orient_net True
python3 run.py
