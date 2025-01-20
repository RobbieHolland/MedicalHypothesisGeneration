#!/bin/bash

# Navigate to the target directory
cd /dataNAS/people/rholland/MedicalHypothesisGeneration || exit 1

# Set environment variables
export PYDEVD_DISABLE_FILE_VALIDATION=1
export PYTHONPATH="/dataNAS/people/rholland/MedicalHypothesisGeneration:/dataNAS/people/akkumar/contrastive-3d"

# Run the Python script using the specified Python environment
/dataNAS/people/rholland/micromamba/envs/venv_torch/bin/python MultimodalPretraining/run/build_vector_database.py
