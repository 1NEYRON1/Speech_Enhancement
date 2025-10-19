from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F

class VoiceBankDataset(Dataset):
    def __init__(self, clean_dir, noisy_dir, max_len=4*16000, mode='train'):
        self.clean_dir = Path(clean_dir)
        self.noisy_dir = Path(noisy_dir)
        self.max_len = max_len
        self.mode = mode
        
        self.clean_files = sorted(list(self.clean_dir.glob("*.wav")))
        self.noisy_files = sorted(list(self.noisy_dir.glob("*.wav")))
        
        if mode == 'test':
            self.file_pairs = []
            clean_dict = {f.name: f for f in self.clean_files}
            for noisy_file in self.noisy_files:
                clean_file = clean_dict.get(noisy_file.name)
                if clean_file:
                    self.file_pairs.append((clean_file, noisy_file))
        else:
            assert len(self.clean_files) == len(self.noisy_files), \
                f"Mismatch: {len(self.clean_files)} clean vs {len(self.noisy_files)} noisy"
            self.file_pairs = list(zip(self.clean_files, self.noisy_files))
        
        print(f"Loaded {len(self.file_pairs)} file pairs for {mode} mode")
    
    def __len__(self):
        return len(self.file_pairs)
    
    def __getitem__(self, idx):
        clean_file, noisy_file = self.file_pairs[idx]
        
        clean_wav, sr = torchaudio.load(clean_file)
        noisy_wav, _ = torchaudio.load(noisy_file)
        
        if clean_wav.shape[0] > 1:
            clean_wav = clean_wav.mean(dim=0, keepdim=True)
        if noisy_wav.shape[0] > 1:
            noisy_wav = noisy_wav.mean(dim=0, keepdim=True)
        
        if self.mode == 'train' and self.max_len:
            if clean_wav.shape[1] > self.max_len:
                start = torch.randint(0, clean_wav.shape[1] - self.max_len, (1,))
                clean_wav = clean_wav[:, start:start+self.max_len]
                noisy_wav = noisy_wav[:, start:start+self.max_len]
            else:
                pad_len = self.max_len - clean_wav.shape[1]
                clean_wav = F.pad(clean_wav, (0, pad_len))
                noisy_wav = F.pad(noisy_wav, (0, pad_len))
        
        return {
            'noisy': noisy_wav.squeeze(0),
            'clean': clean_wav.squeeze(0),
            'filename': clean_file.name,
            'clean_path': str(clean_file),
            'noisy_path': str(noisy_file)
        }