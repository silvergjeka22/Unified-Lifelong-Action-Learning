#  SHARED IMPORTS -> run with: %run src/imports.py
import os
import cv2
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

from tqdm import tqdm
from glob import glob
from PIL import Image
from matplotlib.gridspec import GridSpec

# torch core
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import LSTM
from torch.utils.data import DataLoader, ConcatDataset, Dataset

# torchvision
from torchvision import transforms, models
from torchvision.models import (
    ResNet50_Weights,
    ResNet18_Weights,
    DenseNet121_Weights,
    VGG19_BN_Weights,
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

# vision
subprocess.run(["pip", "install", "torchviz", "-q"], capture_output=True)
from torchviz import make_dot

# project modules
import sys, os as _os
_SRC = _os.path.dirname(_os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from config.config import (
    DATASET_ROOT,
    OUTPUT_ROOT,
    SELECTED_CLASSES,
    FRAME_RATE,
    CLIP_LEN,
    RESIZE_HEIGHT,
    CROP_SIZE,
    TRAIN_SPLIT,
    SEED,
    spatial_transform,
)