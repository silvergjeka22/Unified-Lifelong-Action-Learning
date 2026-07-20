#  SHARED IMPORTS -> run with: %run /content/ulal/src/imports.py
#
# Loads every project symbol into the notebook namespace via %run.
#
# Two bugs fixed in this refactor:
#
#  1. LOAD ORDER. src/meta_learning/ and src/fine_tune/{trainer,visualizer}.py
#     were %run AFTER src/models/ and src/utils/, so the deprecated copies
#     silently overwrote the real ones — 21 name collisions including
#     ResNet50LSTMTeacher, EmbeddingHead, train_model, evaluate_model.
#     Editing src/models/teacher.py had NO effect on any notebook.
#     Those loads are gone; delete the folders when convenient.
#
#  2. DOUBLE CONFIG. `import config.config` (via /content/ulal/src on the path)
#     and `from src.config.config import ...` (via /content/ulal) produced two
#     distinct module objects holding two copies of every value, so mutating
#     cfg.X through one path was invisible through the other. We now put only
#     the repo root on sys.path and import through `src.` exclusively.

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
_ipython = get_ipython()

def _run(rel):
    _ipython.run_line_magic("run", os.path.join(ULAL_ROOT, rel))

# data
_run("src/data/study_dataset.py")
_run("src/data/preprocessing.py")
_run("src/data/dataset.py")
_run("src/data/cache.py")           # feature cache + label space
_run("src/data/youtube.py")

# models
_run("src/models/pretrained.py")
_run("src/models/baselines.py")
_run("src/models/teacher.py")
_run("src/models/student.py")
_run("src/models/head.py")
_run("src/models/temporal_head.py")      # TemporalHead + weight_align

# continual learning
_run("src/cl/ewc.py")
_run("src/cl/rehearsal.py")
_run("src/cl/smart_replay.py")
_run("src/cl/trainer.py")        # train_cl_arm — one loop for every CL arm

# meta-learning  (src/meta/, NOT the deprecated src/meta_learning/)
_run("src/meta/reptile.py")
_run("src/meta/sampler.py")

# knowledge distillation
_run("src/kd/utils.py")
_run("src/kd/trainer.py")

# active domain adaptation
_run("src/active/acquisition.py")
_run("src/active/domain_shift.py")

# GIL GAN
_run("src/gil/gil_gan.py")

# utils  (loaded last so these definitions win)
_run("src/utils/seed.py")
_run("src/utils/metrics.py")
_run("src/utils/probe.py")          # embedding-quality probes
_run("src/utils/visualize.py")
_run("src/utils/save.py")
_run("src/utils/train.py")

# Seed everything before any training happens.
#
# A plain import, not the %run-ed copy above: %run executes each file in its own
# namespace and only copies the resulting symbols into the NOTEBOOK namespace after
# it finishes. So symbols from _run() are not visible inside imports.py itself while
# it is still executing — calling set_seed() here without this import raises
# NameError. The notebooks still get set_seed from the _run above.
from src.utils.seed import set_seed as _set_seed

_set_seed(cfg.SEED)

print(f"ULAL_ROOT : {ULAL_ROOT}")
print(f"Device    : {device}")
if torch.cuda.is_available():
    _p = torch.cuda.get_device_properties(0)
    print(f"GPU       : {_p.name} ({_p.total_memory / 1e9:.1f} GB)")
print(f"Classes   : base={len(cfg.SELECTED_CLASSES)} t1={len(cfg.TASK_1)} "
      f"t2={len(cfg.TASK_2)} t3={len(cfg.TASK_3)} t4={len(cfg.TASK_4)}")
