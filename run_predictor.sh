#!/bin/bash 
#SBATCH --ntasks-per-node=1 # Tasks to be run
#SBATCH --nodes=1  # Run the tasks on the same node
#SBATCH --time=1:00:00   # Required, estimate 5 minutes
#SBATCH --reservation=h100-testing
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:4
#SBATCH --account=ai4wind # Required
#SBATCH --partition=gpu-h100
#SBATCH --exclusive
#SBATCH --error=pred-%j.err
#SBATCH --output=pred%j.out
#SBATCH --mem=100000

echo "Run predictor..."
module load conda

conda activate ldm_omar

python predictor.py  --logdir '/projects/ai4wind/osallam/inflows/DiT/logs_Not_Normalized_sigmoid_vort_x/'  --test_path '/projects/ai4wind/acortiel/inflows/test_data_tiles/test' --num_fr 50 
