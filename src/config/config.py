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

# PREPROCESSING
# Optional cap on TRAIN clips per class - a memory/speed saver (like main's max_samples).
# None = use every clip (full data, best accuracy). Set e.g. 8 for a fast dry-run on Colab.
# Only the train split is trimmed; val/test and the group-disjoint split stay unchanged.
MAX_SAMPLES = None

# TRAINING
# BATCH_SIZE is kept small so every study notebook fits the free-Colab T4 (16 GB).
# Memory scales with BATCH_SIZE * CLIP_LEN images through the ResNet50 conv stack, and
# the replay arms double the batch (new + replayed clips), so 4 is the safe shared value.
# If you still hit CUDA OOM, this is the first knob to lower (3 or 2).
BATCH_SIZE  = 4
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
EWC_LAMBDA         = 2800.0    # re-tune from the CE/EWC print: raise if base forgets, lower if the new task will not learn
REPLAY_BUFFER_SIZE = 300       # replay buffer: ~15 clips per class
LAMBDA_DISTILL     = 3.0       # LwF / replay+LwF distillation weight (raised from 1.0)
KD_TEMPERATURE     = 5.0       # LwF distillation temperature

# KNOWLEDGE DISTILLATION (teacher -> student)
# PyTorch KD tutorial recipe: T=2, loss = 0.75*CE + 0.25*distill, ~10 epochs. Compared:
# CE (no teacher) vs soft-target KD vs cosine vs regressor MSE. NOTE: the tutorial uses lr 1e-3
# because its student is trained FROM SCRATCH; our student's backbone is ImageNet-pretrained, so
# 1e-3 over-writes the pretrained features and tanks accuracy (~60%). Use a fine-tuning lr instead.
KD_T              = 2.0
KD_CE_WEIGHT      = 0.75
KD_DISTILL_WEIGHT = 0.25
KD_EPOCHS         = 15
KD_LR             = 1e-4

# DOMAIN ADAPTATION - YouTube clips (meta-learning hyperparameters live in the notebook)
YT_RAW_DIR = "/content/yt_raw"
# Cap frames read per YouTube video. These clips are minutes long; reading every full-res frame
# into RAM crashes the Colab kernel. 300 frames (~12s at 25 fps) is plenty for a few short clips.
YT_MAX_FRAMES = 300
# Real out-of-domain clips: SEVERAL videos per class so the few-shot task can be VIDEO-DISJOINT
# (adapt on one video, test on the held-out videos). The first URL per class is confirmed working;
# the others are candidates - any dead URL is skipped at download time, and a class needs >=2
# working videos to get a "new video" test. Swap in your own URLs freely.
YOUTUBE_CLIPS = {
    "HorseRiding": [
        "https://archive.org/download/horse-riding_202411/horse%20riding.mp4",
        "https://archive.org/details/lwvtca-Equestrian_Center_October_2024",
    ],
    "Drumming": [
        "https://archive.org/download/Davidleeking-drumming798/Davidleeking-drumming798_512kb.mp4",
        "https://archive.org/details/g678_Drumming_Winter_Concert_12-12-2017_--_ParkTV15",
    ],
    "PushUps": [
        "https://archive.org/download/in-shot-20200102-195128/InShot_20200102_195128.mp4",
        "https://archive.org/details/jccva-Fitness_in_5_Episode_8_-_Advanced_Fitness_Court_Exercises",
    ],
}

random.seed(SEED)

# TRANSFORMS  -  ImageNet normalisation, applied when a clip is preprocessed.
spatial_transform = transforms.Compose([
    transforms.Resize((RESIZE_HEIGHT, RESIZE_HEIGHT)),
    transforms.CenterCrop(CROP_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
