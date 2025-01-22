#!/bin/bash
#SBATCH --partition=BMR-AI
#SBATCH --nodelist=amalfi
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=/dataNAS/people/rholland/MedicalHypothesisGeneration/slurm/output/build_vector_database/%j.out
#SBATCH --error=/dataNAS/people/rholland/MedicalHypothesisGeneration/slurm/output/build_vector_database/%j.err

# Navigate to the target directory
cd /dataNAS/people/rholland/MedicalHypothesisGeneration || exit 1

# Set environment variables
export PYDEVD_DISABLE_FILE_VALIDATION=1
export PYTHONPATH="/dataNAS/people/rholland/MedicalHypothesisGeneration:/dataNAS/people/akkumar/contrastive-3d"

# Run the Python script using the specified Python environment
/dataNAS/people/rholland/micromamba/envs/venv_torch/bin/python MultimodalPretraining/run/build_vector_database.py data=abdominal_ct task=build_vector_database task.vector_database_name=all_25_01_21 task.max_steps=20000