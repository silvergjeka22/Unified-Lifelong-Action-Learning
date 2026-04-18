import os
import cv2
import torch
import random
import shutil
from tqdm import tqdm
from torchvision import transforms
import pandas as pd


import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torch.utils.data import DataLoader, ConcatDataset
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


from torch.utils.data import Dataset
from glob import glob


from torch.nn import LSTM


import torch.nn.functional as F
import itertools


import pickle
from matplotlib.gridspec import GridSpec


# vision
import subprocess
subprocess.run(["pip", "install", "torchviz"], capture_output=True)
from torchviz import make_dot


# weights
from torchvision.models import ResNet50_Weights
from torchvision.models import ResNet18_Weights
from torchvision.models import DenseNet121_Weights
from torchvision.models import VGG19_BN_Weights

# how to run
# %run src/imports.py