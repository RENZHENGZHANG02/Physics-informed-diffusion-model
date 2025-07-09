#!/bin/bash

LOGFILE="graphdit_tmux_$(date +%Y%m%d_%H%M%S).log"

echo "------------------------------------------------" | tee -a $LOGFILE
echo "GraphDiT Job Started: $(date)" | tee -a $LOGFILE
echo "Running on host: $(hostname)" | tee -a $LOGFILE
echo "Working directory: $(pwd)" | tee -a $LOGFILE
echo "------------------------------------------------" | tee -a $LOGFILE

# Properly initialize Conda in non-interactive shell
source /home/xchen5/local/miniconda3/etc/profile.d/conda.sh
conda activate py39

# Use only GPU #3
export CUDA_VISIBLE_DEVICES=3

echo "Python path: $(which python)" | tee -a $LOGFILE
echo "Conda env: $(conda info --envs | grep \* | awk '{print $1}')" | tee -a $LOGFILE

# Run the script
echo "Starting Python script..." | tee -a $LOGFILE
python main.py --config-name=config.yaml \
    model.ensure_connected=True \
    dataset.task_name='heat_related_output' \
    dataset.guidance_target='mu-alpha-zpve-Cv' | tee -a $LOGFILE

echo "------------------------------------------------" | tee -a $LOGFILE
echo "GraphDiT Job Finished: $(date)" | tee -a $LOGFILE
echo "------------------------------------------------" | tee -a $LOGFILE
