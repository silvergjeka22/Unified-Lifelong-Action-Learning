import os
import random
from torchvision import transforms

# PATHS
# Driven by environment variables with sensible defaults. bootstrap.setup() sets
# them on Colab, so this file is never edited at runtime.
DATASET_ROOT = os.environ.get(
    "ULAL_DATASET_ROOT",
    "/root/.cache/kagglehub/datasets/matthewjansen/ucf101-action-recognition/versions/4/",
)
OUTPUT_ROOT = os.environ.get("ULAL_OUTPUT_ROOT", "/content/ucf101-processed/")

# Preprocessed clips live on the Colab local disk (not Drive). Group-disjoint roots,
# one per task, written by preprocess_group_split().
DATA_ROOT  = os.environ.get("ULAL_DATA_ROOT", "/content/UCF101")
BASE_ROOT  = f"{DATA_ROOT}/grouped_base"
TASK1_ROOT = f"{DATA_ROOT}/grouped_task1"
TASK2_ROOT = f"{DATA_ROOT}/grouped_task2"

# Drive: only small things (trained model, result JSONs, plots) go here.
DRIVE_PROJECT = os.environ.get("ULAL_DRIVE_PROJECT", "/content/drive/MyDrive/apai")
CKPT_DIR      = f"{DRIVE_PROJECT}/checkpoints"
RESULTS_DIR   = f"{DRIVE_PROJECT}/results"
RESNET50_PATH = os.environ.get("ULAL_RESNET50_PATH", f"{CKPT_DIR}/ResNet50_10C.pth")

# CLASSES  -  10 base + 5 + 5 = 20, a 10 -> 15 -> 20 continual-learning stream.
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

ALL_CLASSES = SELECTED_CLASSES + TASK_1 + TASK_2

# VIDEO
FRAME_RATE    = 25
CLIP_LEN      = 16
RESIZE_HEIGHT = 256
CROP_SIZE     = 224

# TRAINING
BATCH_SIZE  = 8
SEED        = 42
DROPOUT_P   = 0.4
HIDDEN_SIZE = 256      # LSTM hidden size, shared by teacher and student
TRAIN_SPLIT = 0.8

# CONTINUAL LEARNING - shared training budget
# Every CL arm (naive, EWC, LwF, replay, replay+LwF) trains with THIS lr and epoch
# count, so the only difference between arms is the mechanism, not the step size.
CL_LR     = 1e-4
CL_EPOCHS = 5

# Per-method strength knobs - each arm keeps its own; these are what you tune.
EWC_LAMBDA         = 2000.0    # re-tune from the CE/EWC print: raise if base forgets, lower if the new task will not learn
REPLAY_BUFFER_SIZE = 300       # replay buffer: ~15 clips per class
LAMBDA_DISTILL     = 3.0       # LwF / replay+LwF distillation weight (raised from 1.0)
KD_TEMPERATURE     = 5.0       # LwF distillation temperature

# KNOWLEDGE DISTILLATION (teacher -> student)
KD_T              = 2.0
KD_CE_WEIGHT      = 0.75
KD_DISTILL_WEIGHT = 0.25
KD_EPOCHS         = 10
KD_LR             = 1e-3

# DOMAIN ADAPTATION - YouTube clips (meta-learning hyperparameters live in the notebook)
YT_RAW_DIR = "/content/yt_raw"
YOUTUBE_CLIPS = {
    "HorseRiding": "https://archive.org/download/horse-riding_202411/horse%20riding.mp4",
    "Drumming":    "https://archive.org/download/Davidleeking-drumming798/Davidleeking-drumming798_512kb.mp4",
    "Diving":      "https://archive.org/download/cabeurfm_000005/cabeurfm_000005_access.mp4",
    "PushUps":     "https://archive.org/download/in-shot-20200102-195128/InShot_20200102_195128.mp4",
}

random.seed(SEED)

# TRANSFORMS  -  ImageNet normalisation, applied when a clip is preprocessed.
spatial_transform = transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_HEIGHT)),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
