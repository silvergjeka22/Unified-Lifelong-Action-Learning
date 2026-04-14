import random
import os

# DATASET PATHS
# Auto-updated by setup_colab.sh do not edit manually
DATASET_ROOT = "/root/.cache/kagglehub/datasets/matthewjansen/ucf101-action-recognition/versions/4/"
OUTPUT_ROOT  = "/content/ucf101-processed/"

# VIDEO PROCESSING
FRAME_RATE     = 25     # Target frame rate (fps)
CLIP_LEN       = 16     # Number of frames per clip
RESIZE_HEIGHT  = 256    # Resize video height before cropping
CROP_SIZE      = 224    # Final spatial crop size (224x224)

# DATASET SPLIT
TRAIN_SPLIT = 0.8       # 80% train, 20% test
VAL_SPLIT   = 0.1       # 10% of train used as validation
SEED        = 42        # Global random seed