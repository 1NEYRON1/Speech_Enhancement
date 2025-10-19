from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from pesq import pesq
from pystoi import stoi
import soundfile as sf
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

torch.manual_seed(777)
np.random.seed(777)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

from scripts.training import train_model
from scripts.config import Config

config = Config()

print("=" * 70)
print("MP-SENet with xLSTM+Mamba for Speech Enhancement")
print("=" * 70)

print("\n[1] Training model...")
model, history = train_model(config, device)