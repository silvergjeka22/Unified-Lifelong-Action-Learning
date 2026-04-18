import random
import os
from torchvision import transforms

# DATASET PATHS
# Auto-updated by setup_colab.sh do not edit manually
DATASET_ROOT = "/root/.cache/kagglehub/datasets/matthewjansen/ucf101-action-recognition/versions/4/"
OUTPUT_ROOT  = "/content/ucf101-processed/"

# CLASSES
SELECTED_CLASSES = [
    "PlayingTabla", "PommelHorse", "JumpingJack", "PushUps", "PoleVault",
    "HorseRace", "HighJump", "Drumming", "HorseRiding", "Diving"]
'''
        "BreastStroke", "TrampolineJumping", "YoYo", "SalsaSpin", "WalkingWithDog",
        "VolleyballSpiking", "ThrowDiscus", "TennisSwing", "TaiChi", "Swing",
        "SoccerJuggling", "Skijet", "Skiing", "SkateBoarding", "Rowing",
        "RopeClimbing", "RockClimbingIndoor", "Punch", "PullUps", "PlayingViolin",
        "PlayingPiano", "PlayingGuitar", "HulaHoop", "GolfSwing", "Fencing",
        "CleanAndJerk", "Billiards", "Biking", "BenchPress", "BaseballPitch"
    ]
'''


# VIDEO PROCESSING
FRAME_RATE     = 25     # Target frame rate (fps)
CLIP_LEN       = 16     # Number of frames per clip
RESIZE_HEIGHT  = 256    # Resize video height before cropping
CROP_SIZE      = 224    # Final spatial crop size (224x224)

BATCH_SIZE  = 4

# DATASET SPLIT
TRAIN_SPLIT = 0.8       # 80% train, 20% test
SEED        = 42        # Global random seed

random.seed(SEED)

# TRANSFORMS
spatial_transform = transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_HEIGHT)),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats
        std=[0.229, 0.224, 0.225]
    )
])