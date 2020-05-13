#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:v100:1
#SBATCH --time=22:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=GPUDemo
#SBATCH --mail-type=END
#SBATCH --mail-user=md4289@nyu.edu
#SBATCH --output=monodepth.out

source /scratch/md4289/nlu/env/bin/activate
python train.py --model_name mono_model --log_dir "res-net-50-logs-new" --num_layers 50 --load_weights_folder "res-net-50-logs/mono_model/models/weights_8" > monoresults.out
