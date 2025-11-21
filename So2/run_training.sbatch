#!/bin/bash
#SBATCH --job-name=So2_training
#SBATCH --partition=hgx
#SBATCH --account=mgr_mikro
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G 
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/So2_training_output_%j.out
#SBATCH --error=logs/So2_training_error_%j.err 
 
#Create logs directory 
mkdir -p logs


echo "=== STARTING JOB: $(date) ==="

set -e

echo "Job is running on node: $(hostname)" 


#Conda initialization
source /home/inf151561/miniconda3/etc/profile.d/conda.sh
echo "Conda initialized."

#1: DATA PREPARATION

PERMANENT_TILE_DIR="/home/$USER/sol3/train_tiles_permanent"
RAID_TILE_DIR="/raid/users/$USER/sol3_job_$SLURM_JOB_ID/train_tiles"

echo "Permanent tile storage: $PERMANENT_TILE_DIR"
echo "Temporary /raid tile storage for this job: $RAID_TILE_DIR" 


#Activate the conda env
conda activate mayo_env

 #GPU access check
 echo "Verify PyTorch access GPU"
 python -c "import torch; assert torch.cuda.is_available(), 'PyTorch cannot access CUDA.' ; print('CUDA is available. GPU Name:', torch.cuda.get_device_name(0))"


 #Python cache cleaning
 echo "Clean Python __pycache__ directories" 
 find . -type d -name "__pycache__" -exec rm -r {} +
 echo "Verification successful."

 #Preprocess images only if the directory doesn't exist
 if [ -d "$PERMANENT_TILE_DIR" ] && [ "$(ls -A $PERMANENT_TILE_DIR)" ]; then
    echo "Image directory found. Skip preprocessing."
 else
    echo "Image directory not found or empty. Run prep"
    mkdir -p "$PERMANENT_TILE_DIR"
    python preprocess.py --image_dir ../data/train --csv_path ../data/train.csv --output_dir "$PERMANENT_TILE_DIR" 
 fi


 
 #Copy the permanent tiles to /raid disk
 echo "Copy tiles to raid "
 mkdir -p "$(dirname "$RAID_TILE_DIR")"
 cp -r "$PERMANENT_TILE_DIR" "$RAID_TILE_DIR"
 

 #2: TRAINING

 echo "Run training script, read from $RAID_TILE_DIR"
 python train.py --tile_dir "$RAID_TILE_DIR"
 
 echo "=== JOB FINISHED: $(date) ==="

#Clean up /raid disk
echo "Clean up temp directory: $(dirname "$RAID_TILE_DIR")"
rm -rf "$(dirname "$RAID_TILE_DIR")" 
