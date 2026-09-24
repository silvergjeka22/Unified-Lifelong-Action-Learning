import os
import random
from torchvision import transforms

# PATHS
DATASET_ROOT = os.environ.get(
    "ULAL_DATASET_ROOT",
    "/root/.cache/kagglehub/datasets/matthewjansen/ucf101-action-recognition/versions/4/",
)
OUTPUT_ROOT = os.environ.get("ULAL_OUTPUT_ROOT", "/content/ucf101-processed/")

DATA_ROOT  = os.environ.get("ULAL_DATA_ROOT", "/content/UCF101")
BASE_ROOT  = f"{DATA_ROOT}/grouped_base"
TASK1_ROOT = f"{DATA_ROOT}/grouped_task1"
TASK2_ROOT = f"{DATA_ROOT}/grouped_task2"

# Drive
DRIVE_PROJECT = os.environ.get("ULAL_DRIVE_PROJECT", "/content/drive/MyDrive/apai")
CKPT_DIR      = f"{DRIVE_PROJECT}/checkpoints"
RESULTS_DIR   = f"{DRIVE_PROJECT}/results"
RESNET50_PATH = os.environ.get("ULAL_RESNET50_PATH", f"{CKPT_DIR}/ResNet50_10C.pth")

# CLASSES
SELECTED_CLASSES = [
    "PlayingTabla", "PommelHorse", "JumpingJack", "PushUps", "PoleVault",
    "HorseRace", "HighJump", "Drumming", "HorseRiding", "Diving",
]

TASK_1 = [
    "ApplyEyeMakeup", "ApplyLipstick", "Archery", "BabyCrawling", "BalanceBeam",
]

TASK_2 = [
    "BoxingPunchingBag", "BoxingSpeedBag", "BrushingTeeth", "CliffDiving", "CricketBowling",
]

TASK_3 = [
    "BandMarching", "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress",
]

TASK_4 = [
    "Biking", "Billiards", "BlowDryHair", "BlowingCandles", "BodyWeightSquats",
]

TASK_5 = [
    "Bowling", "BreastStroke", "CleanAndJerk", "CuttingInKitchen", "Fencing",
]

TASK_6 = [
    "FieldHockeyPenalty", "FloorGymnastics", "FrisbeeCatch", "FrontCrawl", "GolfSwing",
]

MUSIC = [
    "PlayingCello", "PlayingDaf", "PlayingDhol", "PlayingFlute",
    "PlayingGuitar", "PlayingPiano", "PlayingSitar", "PlayingViolin",
]

FINEGRAIN = [
    "Lunges", "WallPushups", "HandstandPushups", "PullUps", "Nunchucks", "JumpRope",
    "TaiChi", "ShavingBeard", "Haircut", "HeadMassage", "MoppingFloor", "HandstandWalking",
]

# sports MAML meta trains
SPORTS_META = [
    "BaseballPitch", "Basketball", "BasketballDunk", "BenchPress", "Biking",
    "Billiards", "BodyWeightSquats", "Bowling", "BreastStroke", "CleanAndJerk",
    "CricketShot", "Fencing", "FieldHockeyPenalty", "FloorGymnastics", "FrisbeeCatch",
    "HulaHoop", "IceDancing", "JavelinThrow", "JugglingBalls", "JumpRope",
    "Kayaking", "Lunges", "Nunchucks", "ParallelBars", "PullUps",
    "TableTennisShot", "TennisSwing", "SoccerPenalty", "VolleyballSpiking", "LongJump",
    "Skiing", "Surfing", "ThrowDiscus", "TrampolineJumping", "SumoWrestling",
    "Punch", "Rafting", "RockClimbingIndoor", "RopeClimbing", "Rowing"
]

# NEW MAML EXAM
SPORTS_EXAM = [
    #"Shotput", "SkateBoarding", "Skijet", "SkyDiving", "SoccerJuggling"
    "FrontCrawl", "GolfSwing", "HammerThrow", "HandstandPushups", "HandstandWalking",
]

ALL_CLASSES = SELECTED_CLASSES + TASK_1 + TASK_2 + TASK_3 + TASK_4 + TASK_5 + TASK_6 + MUSIC + FINEGRAIN

# YOUTUBE
YT_RAW_DIR = "/content/yt_raw"

# VIDEO
FRAME_RATE    = 25
CLIP_LEN      = 16
RESIZE_HEIGHT = 256
CROP_SIZE     = 224

# PREPROCESSING
MAX_SAMPLES = None

# TRAINING
BATCH_SIZE  = 4
SEED        = 42
DROPOUT_P   = 0.4
HIDDEN_SIZE = 256      
TRAIN_SPLIT = 0.8

# CONTINUAL LEARNING
CL_LR     = 1e-4
CL_EPOCHS = 5

# EWC & LWF
EWC_LAMBDA         = 2800.0
REPLAY_BUFFER_SIZE = 300       
LAMBDA_DISTILL     = 3.0      
KD_TEMPERATURE     = 5.0       

# KNOWLEDGE DISTILLATION
KD_T              = 2.0
KD_CE_WEIGHT      = 0.75
KD_DISTILL_WEIGHT = 0.25
KD_EPOCHS         = 15
KD_LR             = 1e-4

random.seed(SEED)

# TRANSFORMS
spatial_transform = transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_HEIGHT)),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
