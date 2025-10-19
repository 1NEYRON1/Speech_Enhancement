import os
import torch
import torchaudio
from torch.utils.data import Dataset
import numpy as np
import torch.nn.functional as F

class VoiceBankDemandDataset(Dataset):
    def __init__(self, data_dir, is_train=True, segment_length_sec=4):
        """
        Args:
            data_dir (str): Путь к корневой папке датасета (например, './data/').
            is_train (bool): True для тренировочного набора, False для тестового.
            segment_length_sec (int): Длина сегмента аудио для обучения в секундах.
        """
        self.is_train = is_train
        
        if self.is_train:
            self.clean_dir = os.path.join(data_dir, "wav_clean_train")
            self.noisy_dir = os.path.join(data_dir, "wav_noisy_train")
        else:
            self.clean_dir = os.path.join(data_dir, "wav_clean_test")
            self.noisy_dir = os.path.join(data_dir, "wav_noisy_test")

        self.file_list = [f for f in os.listdir(self.noisy_dir) if f.endswith('.wav')]
        
        # Параметры для STFT
        self.n_fft = 512
        self.hop_length = 256
        self.win_length = 512
        self.window = torch.hann_window(self.win_length)
        
        # Параметры аудио
        self.sample_rate = 16000
        self.segment_length = self.sample_rate * segment_length_sec

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        
        clean_path = os.path.join(self.clean_dir, file_name)
        noisy_path = os.path.join(self.noisy_dir, file_name)
        
        # Загружаем аудио
        clean_wav, sr_c = torchaudio.load(clean_path)
        noisy_wav, sr_n = torchaudio.load(noisy_path)
        
        assert sr_c == self.sample_rate and sr_n == self.sample_rate

        # Приводим к моно
        if clean_wav.shape[0] > 1:
            clean_wav = torch.mean(clean_wav, dim=0, keepdim=True)
        if noisy_wav.shape[0] > 1:
            noisy_wav = torch.mean(noisy_wav, dim=0, keepdim=True)

        # Вырезаем случайный сегмент для обучения
        if self.is_train:
            if clean_wav.shape[1] > self.segment_length:
                start = torch.randint(0, clean_wav.shape[1] - self.segment_length, (1,)).item()
                clean_wav = clean_wav[:, start:start + self.segment_length]
                noisy_wav = noisy_wav[:, start:start + self.segment_length]
            else: # Если файл короче, дополняем нулями
                clean_wav = F.pad(clean_wav, (0, self.segment_length - clean_wav.shape[1]))
                noisy_wav = F.pad(noisy_wav, (0, self.segment_length - noisy_wav.shape[1]))

        # Выполняем STFT
        clean_stft = torch.stft(clean_wav.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length,
                                win_length=self.win_length, window=self.window, return_complex=True)
        noisy_stft = torch.stft(noisy_wav.squeeze(0), n_fft=self.n_fft, hop_length=self.hop_length,
                                win_length=self.win_length, window=self.window, return_complex=True)

        # Получаем амплитуду и фазу
        clean_amp = torch.abs(clean_stft)
        clean_pha = torch.angle(clean_stft)
        
        noisy_amp = torch.abs(noisy_stft)
        noisy_pha = torch.angle(noisy_stft)
        
        return noisy_amp, noisy_pha, clean_amp, clean_pha, file_name
