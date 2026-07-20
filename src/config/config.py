import random
import os
from torchvision import transforms

# DATASET PATHS
# Driven by environment variables — set them before importing, or accept the
# defaults. bootstrap.setup_dataset() sets them automatically on Colab.
#
# Nothing rewrites this file at runtime any more: the old setup_colab.sh used
# sed to patch DATASET_ROOT/OUTPUT_ROOT in place and then forced a kernel
# restart. That made the file differ from git, and a re-clone silently reverted
# your paths. Env vars keep the file immutable.
DATASET_ROOT = os.environ.get(
    "ULAL_DATASET_ROOT",
    "/root/.cache/kagglehub/datasets/matthewjansen/ucf101-action-recognition/versions/4/",
)
OUTPUT_ROOT  = os.environ.get("ULAL_OUTPUT_ROOT", "/content/ucf101-processed/")

# processed data paths — kept under DATA_ROOT so they survive a repo re-clone
DATA_ROOT  = os.environ.get("ULAL_DATA_ROOT", "/content/UCF101")
BASE_ROOT  = f"{DATA_ROOT}/processed_data_base"
TASK1_ROOT = f"{DATA_ROOT}/processed_data_task1"
TASK2_ROOT = f"{DATA_ROOT}/processed_data_task2"
TASK3_ROOT = f"{DATA_ROOT}/processed_data_task3"
TASK4_ROOT = f"{DATA_ROOT}/processed_data_task4"

# ── Drive paths ───────────────────────────────────────────────────────────────
DRIVE_PROJECT = os.environ.get("ULAL_DRIVE_PROJECT", "/content/drive/MyDrive/apai")

# resnet50 path (Section 1 output)
RESNET50_PATH = os.environ.get(
    "ULAL_RESNET50_PATH", f"{DRIVE_PROJECT}/resnet50/models/ResNet50_10C.pth"
)
# (CKPT_DIR / RESULTS_DIR / CACHE_DIR / GIL_CKPT_DIR defined below, after DATA_ROOT)

# CLASSES
SELECTED_CLASSES = [
    "PlayingTabla", "PommelHorse", "JumpingJack", "PushUps", "PoleVault", "HorseRace", "HighJump", "Drumming", "HorseRiding", "Diving"]
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
BATCH_SIZE  = 8
NUM_WORKERS = 2

# DATASET SPLIT
TRAIN_SPLIT = 0.8       # 80% train, 20% test
SEED        = 42        # Global random seed

# DROPOUT
DROPOUT_P = 0.4

random.seed(SEED)

# ── CL experiment paths ───────────────────────────────────────────────────────
EXEMPLAR_ROOT        = f"{DATA_ROOT}/processed_exemplars"
SUBSET1_ROOT         = f"{DATA_ROOT}/processed_subset1"
LIMITED_SUBSET1_ROOT = f"{DATA_ROOT}/processed_limited_subset1"
SUBSET2_ROOT         = f"{DATA_ROOT}/processed_subset2"
EXEMPLAR_LIMIT       = 5     # max train samples per class for exemplar sets

# ── Derived Drive paths (Section outputs) ────────────────────────────────────
CKPT_DIR     = f"{DRIVE_PROJECT}/checkpoints"
RESULTS_DIR  = f"{DRIVE_PROJECT}/results"
CACHE_DIR    = f"{DRIVE_PROJECT}/cache"
GIL_CKPT_DIR = f"{DRIVE_PROJECT}/gil"
FEAT_CACHE   = f"{CACHE_DIR}/ulal_frame_features.pt"

# ── Frame-feature caching (Section 2) ────────────────────────────────────────
BACKBONE_DIM = 2048          # ResNet50 pooled output per frame
STORE_DTYPE  = "float16"     # on-disk dtype for cached features

# ── Teacher / Student architecture ───────────────────────────────────────────
TEACHER_HIDDEN  = 256
STUDENT_HIDDEN  = 128
HEAD_DROPOUT    = 0.25
STUDENT_DROPOUT = 0.20

# ── EWC ──────────────────────────────────────────────────────────────────────
EWC_LAMBDA = 5000.0
EWC_LR     = 1e-5
EWC_WD     = 1e-4
EWC_EPOCHS = 5

# ── Replay / LwF ─────────────────────────────────────────────────────────────
LAMBDA_DISTILL = 2.0
REPLAY_EPOCHS  = 5
REPLAY_LR      = 1e-4

# ── Knowledge Distillation ───────────────────────────────────────────────────
CE_WEIGHT  = 1.0
MSE_WEIGHT = 0.2

# ── Reptile / Meta-learning ──────────────────────────────────────────────────
K_SHOT             = 40
K_QUERY            = 1
K_SUPPORT          = 3
INNER_LR           = 0.0005
INNER_STEPS        = 10
SRC_EPSILON        = 0.20
REPTILE_EPOCHS     = 25
EPISODES_PER_EPOCH = 20
REPTILE_PATIENCE   = 5

# ── Fine-tune head ────────────────────────────────────────────────────────────
FINETUNE_EPOCHS   = 15
FINETUNE_LR       = 0.0005
FINETUNE_WD       = 0.03
FINETUNE_BATCH    = 12
FINETUNE_PATIENCE = 8
LABEL_SMOOTHING   = 0.10

# ── Active Domain Adaptation ─────────────────────────────────────────────────
AL_BUDGET_PER_CLASS = 10
AL_STRATEGY         = "entropy"
ADAPT_INNER_LR      = 0.0005
ADAPT_INNER_STEPS   = 10
ADAPT_EPSILON       = 0.20
ADAPT_EPISODES      = 30
CLIPS_PER_VID       = 8
YT_RAW_DIR          = "/content/yt_raw"
YT_CLIPS_DIR        = "/content/yt_clips"

# YouTube videos {class_name: url} — edit to change which classes are tested
YOUTUBE_CLIPS = {
    "Diving":      "https://www.youtube.com/watch?v=sM4w7GgJjNs",
    "HorseRiding": "https://www.youtube.com/watch?v=dMoWGWA_sFk",
    "PushUps":     "https://www.youtube.com/watch?v=IODxDxX7oi4",
}

# ── GAN / GIL ────────────────────────────────────────────────────────────────
FEAT_DIM         = 256
LATENT_DIM       = 256
NOISE_DIM        = 256
SEM_DIM          = 384
GAN_EPOCHS       = 30
GAN_LR           = 1e-4
LAM1             = 0.01
LAM2             = 0.1
ALPHA_GP         = 10.0
N_CRITIC         = 5
FT_EPOCHS        = 5
FT_LR            = 1e-4
BATCH_SIZE_GAN   = 64
J_SYNTH          = 50
CVAE_LR          = 1e-4
CVAE_EPOCHS      = 5
CVAE_INIT_EPOCHS = 50

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