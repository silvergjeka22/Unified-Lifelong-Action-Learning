"""
Reproducibility.

Without this, every run differs because these are all unseeded:

    nn.Linear init      the classifier and projector are freshly random each run
    expand_head         new class rows are random
    nn.Dropout          different mask every forward pass
    DataLoader(shuffle) different batch order
    buffer.sample_batch torch.randperm picks different exemplars
    cuDNN autotuning    benchmark=True may select different algorithms

config.py calls random.seed(SEED), but that only covers Python's `random` module —
it does nothing for torch or numpy, which is where all of the above live.

On ~300 training clips per task evaluated on ~55 test clips, that easily moves a
result by 5-15 points between runs.
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True, verbose: bool = False):
    """
    Seed every RNG the pipeline touches.

    Call once during setup AND again immediately before each training run, so a
    result does not depend on how many cells you happened to execute beforehand.

    deterministic=True also pins cuDNN. That costs a little speed but makes an LSTM
    run repeatable; benchmark=True (the fast path) can pick different convolution
    algorithms between runs and change results slightly.

    Note: for full cuDNN RNN determinism, CUBLAS_WORKSPACE_CONFIG must be set BEFORE
    CUDA initialises. The notebooks set it in the bootstrap cell, which runs before
    torch is imported.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    if verbose:
        print(f"seed={seed} deterministic={deterministic}")
    return seed


def seed_worker(worker_id):
    """DataLoader worker_init_fn — only needed when num_workers > 0."""
    s = torch.initial_seed() % 2**32
    np.random.seed(s)
    random.seed(s)


def loader_generator(seed: int = 42):
    """Pass as DataLoader(generator=...) to pin shuffle order independently."""
    g = torch.Generator()
    g.manual_seed(seed)
    return g
