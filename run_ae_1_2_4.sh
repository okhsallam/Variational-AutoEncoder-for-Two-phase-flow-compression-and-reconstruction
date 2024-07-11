#!/bin/bash 
#SBATCH --ntasks-per-node=1 # Tasks to be run
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --time=40:00:00   # Required, estimate 5 minutes
#SBATCH --reservation=h100-testing
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:1
#SBATCH --account=ai4wind # Required
#SBATCH --partition=gpu-h100
#SBATCH --exclusive
#SBATCH --error=ae-%j.err
#SBATCH --output=ae-%j.out
#SBATCH --mem=100000

echo "Training AE..."
module load conda

conda activate ldm_omar

#CUDA_VISIBLE_DEVICES=0, python main.py --base configs/ae_1_2_4.yaml -t --gpus 0, --logdir /projects/ai4wind/osallam/inflows/DiT/logs_Github

CUDA_VISIBLE_DEVICES=0, python main.py --base configs/ae_1_2_4.yaml -t --gpus 0,

