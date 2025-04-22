#!/bin/bash
#SBATCH --job-name=ECGGCN
#SBATCH --output=log_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ../../../python_venv/CS584/bin/activate
echo "Running data preprocessing script..."
python 01_organize_data.py
echo "Running training script..."
python 02_train.py
echo "Running plotting script..."
python 03_plot_metrics.py
