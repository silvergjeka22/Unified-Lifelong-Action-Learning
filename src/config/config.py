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
        # if we will use 20
        ["BreastStroke", "TrampolineJumping", "YoYo", "SalsaSpin", "WalkingWithDog",
        "VolleyballSpiking", "ThrowDiscus", "TennisSwing", "TaiChi", "Swing",]
'''

TASK_1 = [
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "BlowDryHair",
    "BlowingCandles",
    "BodyWeightSquats",
    "Bowling",
]

CLASSES_20 = SELECTED_CLASSES + TASK_1

TASK_2 = [
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "BrushingTeeth",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "CuttingInKitchen",
    "FieldHockeyPenalty",
    "Haircut",
    "SoccerPenalty",
]

CLASSES_30 = CLASSES_20 + TASK_2

TASK_3 = [
    "SoccerJuggling",
    "Skijet",
    "Skiing",
    "SkateBoarding",
    "Rowing",
    "RopeClimbing",
    "RockClimbingIndoor",
    "Punch",
    "PullUps",
    "PlayingViolin"
]

CLASSES_40 = CLASSES_30 + TASK_3

TASK_4 = [
    "PlayingPiano",
    "PlayingGuitar",
    "HulaHoop",
    "GolfSwing",
    "Fencing",
    "CleanAndJerk",
    "Billiards",
    "Biking",
    "BenchPress",
    "BaseballPitch"
]

CLASSES_50 = CLASSES_40 + TASK_4

# VIDEO PROCESSING
FRAME_RATE     = 25     # Target frame rate (fps)
CLIP_LEN       = 16     # Number of frames per clip
RESIZE_HEIGHT  = 256    # Resize video height before cropping
CROP_SIZE      = 224    # Final spatial crop size (224x224)

# TRAINING
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