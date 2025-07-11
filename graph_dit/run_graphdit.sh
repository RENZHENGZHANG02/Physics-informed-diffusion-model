#!/bin/bash

LOGFILE="graphdit_tmux_$(date +%Y%m%d_%H%M%S).log"
export CUDA_LAUNCH_BLOCKING=1
export HYDRA_FULL_ERROR=1

echo "------------------------------------------------" | tee -a $LOGFILE
echo "GraphDiT Job Started: $(date)" | tee -a $LOGFILE
echo "Running on host: $(hostname)" | tee -a $LOGFILE
echo "Working directory: $(pwd)" | tee -a $LOGFILE
echo "------------------------------------------------" | tee -a $LOGFILE

# Properly initialize Conda in non-interactive shell
source /home/xchen5/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate py39

# Use only GPU #3
export CUDA_VISIBLE_DEVICES=1

echo "Python path: $(which python)" | tee -a $LOGFILE
echo "Conda env: $(conda info --envs | grep \* | awk '{print $1}')" | tee -a $LOGFILE

# Generate evaluated tasks and related properties
echo "Generate .yaml file for inputs..." | tee -a $LOGFILE
# Notice properties used in evaluation could be different from properties used in guidance
python generate_registry.py \
  --task_dict '{"classification_dataset":["Cv", "gap_class"]}' \
  --task_types '{"classification_dataset":["regression", "classification"]}' | tee -a $LOGFILE

# Run the script
echo "Starting Python script..." | tee -a $LOGFILE
python main.py --config-name=config.yaml \
    model.ensure_connected=True \
    dataset.task_name='classification_dataset' \
    dataset.guidance_target='Cv-gap_class' | tee -a $LOGFILE

echo "------------------------------------------------" | tee -a $LOGFILE
echo "GraphDiT Job Finished: $(date)" | tee -a $LOGFILE
echo "------------------------------------------------" | tee -a $LOGFILE
