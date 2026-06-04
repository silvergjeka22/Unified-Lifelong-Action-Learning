#  SHARED IMPORTS -> run with: %run /content/src/imports.py
import sys
sys.path.insert(0, '/content')
sys.path.insert(0, '/content/src')

import os
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
subprocess.run(["pip", "install", "torchinfo", "-q"], capture_output=True)
from torchinfo import summary

# config
import config.config as cfg
from config.config import (
    DATASET_ROOT,
    OUTPUT_ROOT,
    BASE_ROOT, TASK1_ROOT, TASK2_ROOT, TASK3_ROOT, TASK4_ROOT,
    SELECTED_CLASSES, TASK_1, TASK_2, TASK_3, TASK_4,
    CLASSES_20, CLASSES_30, CLASSES_40, CLASSES_50,
    BATCH_SIZE, NUM_WORKERS,
    FRAME_RATE, CLIP_LEN, RESIZE_HEIGHT, CROP_SIZE,
    TRAIN_SPLIT, SEED,
    DROPOUT_P,
    spatial_transform,
    RESNET50_PATH,
    DRIVE_PROJECT, CKPT_DIR, RESULTS_DIR, GIL_CKPT_DIR,
    EXEMPLAR_ROOT, SUBSET1_ROOT, LIMITED_SUBSET1_ROOT, SUBSET2_ROOT,
    EXEMPLAR_LIMIT,
    TEACHER_HIDDEN, STUDENT_HIDDEN, HEAD_DROPOUT, STUDENT_DROPOUT,
    EWC_LAMBDA, EWC_LR, EWC_WD, EWC_EPOCHS,
    LAMBDA_DISTILL, REPLAY_EPOCHS, REPLAY_LR,
    CE_WEIGHT, MSE_WEIGHT,
    K_SHOT, K_QUERY, K_SUPPORT,
    INNER_LR, INNER_STEPS, SRC_EPSILON,
    REPTILE_EPOCHS, EPISODES_PER_EPOCH, REPTILE_PATIENCE,
    FINETUNE_EPOCHS, FINETUNE_LR, FINETUNE_WD, FINETUNE_BATCH,
    FINETUNE_PATIENCE, LABEL_SMOOTHING,
    AL_BUDGET_PER_CLASS, AL_STRATEGY,
    ADAPT_INNER_LR, ADAPT_INNER_STEPS, ADAPT_EPSILON, ADAPT_EPISODES,
    CLIPS_PER_VID, YT_RAW_DIR, YT_CLIPS_DIR, YOUTUBE_CLIPS,
    FEAT_DIM, LATENT_DIM, NOISE_DIM, SEM_DIM,
    GAN_EPOCHS, GAN_LR, LAM1, LAM2, ALPHA_GP, N_CRITIC,
    FT_EPOCHS, FT_LR, BATCH_SIZE_GAN, J_SYNTH,
    CVAE_LR, CVAE_EPOCHS, CVAE_INIT_EPOCHS,
)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# project modules (executed via %run so all symbols land in notebook namespace)
_ipython = get_ipython()

_ipython.run_line_magic('run', '/content/src/data/study_dataset.py')
_ipython.run_line_magic('run', '/content/src/data/preprocessing.py')
_ipython.run_line_magic('run', '/content/src/data/dataset.py')
_ipython.run_line_magic('run', '/content/src/data/youtube.py')

_ipython.run_line_magic('run', '/content/src/fine_tune/trainer.py')
_ipython.run_line_magic('run', '/content/src/fine_tune/visualizer.py')

_ipython.run_line_magic('run', '/content/src/models/pretrained.py')
_ipython.run_line_magic('run', '/content/src/models/baselines.py')
_ipython.run_line_magic('run', '/content/src/models/teacher.py')
_ipython.run_line_magic('run', '/content/src/models/student.py')
_ipython.run_line_magic('run', '/content/src/models/head.py')

_ipython.run_line_magic('run', '/content/src/cl/ewc.py')
_ipython.run_line_magic('run', '/content/src/cl/rehearsal.py')
_ipython.run_line_magic('run', '/content/src/cl/smart_replay.py')
_ipython.run_line_magic('run', '/content/src/cl/naive.py')

_ipython.run_line_magic('run', '/content/src/meta/reptile.py')
_ipython.run_line_magic('run', '/content/src/meta/sampler.py')

_ipython.run_line_magic('run', '/content/src/meta_learning/models.py')
_ipython.run_line_magic('run', '/content/src/meta_learning/sampler.py')
_ipython.run_line_magic('run', '/content/src/meta_learning/reptile.py')

_ipython.run_line_magic('run', '/content/src/kd/utils.py')
_ipython.run_line_magic('run', '/content/src/kd/trainer.py')
_ipython.run_line_magic('run', '/content/src/kd/mseLoss.py')

_ipython.run_line_magic('run', '/content/src/active/acquisition.py')
_ipython.run_line_magic('run', '/content/src/active/domain_shift.py')

_ipython.run_line_magic('run', '/content/src/fine_tune/gil_gan.py')

_ipython.run_line_magic('run', '/content/src/utils/metrics.py')
_ipython.run_line_magic('run', '/content/src/utils/visualize.py')
_ipython.run_line_magic('run', '/content/src/utils/save.py')
_ipython.run_line_magic('run', '/content/src/utils/train.py')
