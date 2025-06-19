#!/bin/bash
#SBATCH --job-name=job_name
#SBATCH --time=48:00:00
#SBATCH --account=plgmpr25-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --cpus-per-task=16
#SBATCH --mem=50G
#SBATCH --gres=gpu
#SBATCH --output="joblog-%j.txt"
#SBATCH --error="joberr-%j.txt"
#SBATCH --gres=gpu:8

module purge
module load GCC/11.3.0
module load GCCcore/10.3.0
module load Python/3.9.5
module load OpenMPI/4.1.4
module load PyTorch/1.13.1-CUDA-11.7.0
pip install torchvision
pip install pandas

srun python train.py --output_dir run_10_verylast --resume_epoch 129 --test --root_dir DiagSet-A --magnification 10x --batch_size 1024  --weight_decay 1e-5 --epochs 1000 --freeze_schedule "[[129,2,1e-7]]"
