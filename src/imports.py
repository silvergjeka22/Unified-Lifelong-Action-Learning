# SHARED IMPORTS -> %run /content/ulal/src/imports.py
# Loads every project symbol into the notebook namespace. The repo root goes on
# sys.path and config is imported through src, so there is one cfg object.

import os
import sys

ULAL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ULAL_ROOT not in sys.path:
    sys.path.insert(0, ULAL_ROOT)

# Free-Colab GPU: let the CUDA allocator grow segments instead of fragmenting. Must be
# set before torch initialises CUDA, so it goes above the torch import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
import copy
import json
import torch
import random
import subprocess
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from glob import glob
from PIL import Image

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset, Dataset, TensorDataset

from torchvision import transforms, models
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

subprocess.run([sys.executable, "-m", "pip", "install", "torchinfo", "-q"], capture_output=True)
from torchinfo import summary

# config - one import root
import src.config.config as cfg
from src.config.config import *          # noqa: F403

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# project modules - %run so every symbol lands in the notebook namespace
ipython = get_ipython()

def run_module(rel):
    ipython.run_line_magic("run", os.path.join(ULAL_ROOT, rel))

run_module("src/data/study.py")
run_module("src/data/preprocessing.py")
run_module("src/data/dataset.py")

run_module("src/models/backbones.py")
run_module("src/models/student.py")

run_module("src/training/seed.py")
run_module("src/training/train.py")
run_module("src/training/metrics.py")
run_module("src/training/visualize.py")

run_module("src/cl/ewc.py")
run_module("src/cl/rehearsal.py")

run_module("src/kd/distill.py")

run_module("src/da/youtube.py")

run_module("src/meta/episodic.py")
run_module("src/meta/fscil.py")

from src.training.seed import set_seed
set_seed(cfg.SEED)

print(f"ULAL_ROOT : {ULAL_ROOT}")
print(f"Device    : {device}")
print(f"Classes   : base={len(cfg.SELECTED_CLASSES)} task1={len(cfg.TASK_1)} task2={len(cfg.TASK_2)}")
