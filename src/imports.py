# SHARED IMPORTS -> %run /content/ulal/src/imports.py
# Loads every project symbol into the notebook namespace. Repo root goes on
# sys.path and config is imported through src. only, so there is one cfg object.

import os
import sys

# Repo root = parent of this file's directory. Works wherever the repo is cloned.
ULAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ULAL_ROOT not in sys.path:
    sys.path.insert(0, ULAL_ROOT)

import cv2
import copy
import json
import torch
import random
import shutil
import pickle
import itertools
import subprocess
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from tqdm import tqdm
from glob import glob
from PIL import Image
from matplotlib.gridspec import GridSpec

# torch core
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import LSTM
from torch.utils.data import DataLoader, ConcatDataset, Dataset, TensorDataset

# torchvision
from torchvision import transforms, models
from torchvision.models import (
    ResNet50_Weights,
    ResNet18_Weights,
    DenseNet121_Weights,
    VGG19_BN_Weights,
    MobileNet_V3_Small_Weights,
)

# sklearn
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# torchinfo
subprocess.run([sys.executable, "-m", "pip", "install", "torchinfo", "-q"], capture_output=True)
from torchinfo import summary

# ── config — ONE import root ─────────────────────────────────────────────────
import src.config.config as cfg
from src.config.config import *          # noqa: F403  (paths, class splits, hyperparams)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── project modules (%run so all symbols land in the notebook namespace) ─────
ipython = get_ipython()

def run_module(rel):
    ipython.run_line_magic("run", os.path.join(ULAL_ROOT, rel))

# data
run_module("src/data/study_dataset.py")
run_module("src/data/preprocessing.py")
run_module("src/data/dataset.py")
run_module("src/data/cache.py")           # feature cache + label space
run_module("src/data/youtube.py")

# models
run_module("src/models/pretrained.py")
run_module("src/models/baselines.py")
run_module("src/models/teacher.py")
run_module("src/models/student.py")
run_module("src/models/temporal_head.py")      # TemporalHead + weight_align

# continual learning
run_module("src/cl/ewc.py")
run_module("src/cl/rehearsal.py")
run_module("src/cl/smart_replay.py")
run_module("src/cl/trainer.py")        # train_cl_arm — one loop for every CL arm

# knowledge distillation
run_module("src/kd/utils.py")
run_module("src/kd/trainer.py")

# active domain adaptation  (Reptile adapt lives here, self-contained)
run_module("src/active/acquisition.py")
run_module("src/active/domain_shift.py")

# meta-learning  (Reptile vs first-order MAML)
run_module("src/meta/metalearn.py")

# utils  (loaded last so these definitions win)
run_module("src/utils/seed.py")
run_module("src/utils/metrics.py")
run_module("src/utils/probe.py")          # embedding-quality probes
run_module("src/utils/evaluate.py")       # per-task + real-image evaluation
run_module("src/utils/visualize.py")
run_module("src/utils/save.py")
run_module("src/utils/train.py")

# Imported directly (not the run_module copy): run_module symbols are not visible
# inside this file until it finishes, so set_seed would be undefined here otherwise.
from src.utils.seed import set_seed as seed_all

seed_all(cfg.SEED)

print(f"ULAL_ROOT : {ULAL_ROOT}")
print(f"Device    : {device}")
if torch.cuda.is_available():
    gpu_props = torch.cuda.get_device_properties(0)
    print(f"GPU       : {gpu_props.name} ({gpu_props.total_memory / 1e9:.1f} GB)")
print(f"Classes   : base={len(cfg.SELECTED_CLASSES)} t1={len(cfg.TASK_1)} "
      f"t2={len(cfg.TASK_2)} t3={len(cfg.TASK_3)} t4={len(cfg.TASK_4)}")
