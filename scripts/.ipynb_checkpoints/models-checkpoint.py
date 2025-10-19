import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

class STFT(nn.Module):
    def __init__(self, n_fft=512, hop_length=128, win_length=512):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = torch.hann_window(win_length)

    def forward(self, x):
        self.window = self.window.to(x.device)
        spec = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        return spec

    def inverse(self, spec):
        self.window = self.window.to(spec.device)
        wav = torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
        )
        return wav


class xLSTMBlock(nn.Module):
    """
    Упрощенная реализация xLSTM блока для бейзлайна
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0,
            bidirectional=True,
        )
        self.norm = nn.LayerNorm(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim * 2, input_dim)

    def forward(self, x):
        B, T, n_freq, C = x.shape

        # Применяем LSTM по временной оси
        x_reshaped = x.permute(0, 2, 1, 3).reshape(B * n_freq, T, C)
        out, _ = self.lstm(x_reshaped)
        out = self.norm(out)
        out = self.dropout(out)
        out = self.proj(out)
        out = out.reshape(B, n_freq, T, C).permute(0, 2, 1, 3)

        return out + x  # Residual connection


class MambaBlock(nn.Module):
    """
    Упрощенная реализация Mamba блока для бейзлайна
    """

    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # Используем Conv1D как аппроксимацию SSM
        self.conv = nn.Conv1d(
            input_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=input_dim // 4 if input_dim >= 4 else 1,
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(hidden_dim, input_dim)
        self.activation = nn.SiLU()

    def forward(self, x):
        B, T, n_freq, C = x.shape

        # Применяем по частотной оси
        x_reshaped = x.permute(0, 1, 3, 2).reshape(B * T, C, n_freq)
        out = self.conv(x_reshaped)
        out = out.permute(0, 2, 1).reshape(B, T, n_freq, -1)
        out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.proj(out)

        return out + x  # Residual connection

class HybridBlock(nn.Module):
    """
    Комбинация xLSTM (для времени) и Mamba (для частоты)
    """
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.xlstm = xLSTMBlock(input_dim, hidden_dim, dropout)
        self.mamba = MambaBlock(input_dim, hidden_dim, dropout)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        # x: [B, T, F, C]
        x = self.xlstm(x)
        x = self.mamba(x)
        x = self.norm(x)
        return x

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            HybridBlock(hidden_dim, hidden_dim, dropout)
            for _ in range(num_layers)
        ])
        
    def forward(self, x):
        x = self.input_proj(x)
        for layer in self.layers:
            x = layer(x)
        return x

class MagnitudeDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [HybridBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        return x

class PhaseDecoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [HybridBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_layers)]
        )
        self.output_proj = nn.Sequential(nn.Linear(hidden_dim, output_dim), nn.Tanh())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_proj(x)
        return x * np.pi  # Масштабируем в [-pi, pi]

class MPSENet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.stft = STFT(
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
        )

        freq_dim = config.n_fft // 2 + 1

        self.encoder = Encoder(
            input_dim=2,  # magnitude + phase
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout,
        )

        # Параллельные декодеры
        self.magnitude_decoder = MagnitudeDecoder(
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_layers=config.num_layers // 2,
            dropout=config.dropout,
        )

        self.phase_decoder = PhaseDecoder(
            hidden_dim=config.hidden_dim,
            output_dim=1,
            num_layers=config.num_layers // 2,
            dropout=config.dropout,
        )

    def forward(self, noisy_wav):
        # Запоминаем исходную длину
        original_length = noisy_wav.shape[-1]
        
        # STFT
        noisy_spec = self.stft(noisy_wav)
        noisy_mag = torch.abs(noisy_spec)
        noisy_phase = torch.angle(noisy_spec)
        
        # B, n_freq, T = noisy_mag.shape
        
        # Подготовка входа
        encoder_input = torch.stack([noisy_mag, noisy_phase], dim=-1)
        encoder_input = encoder_input.permute(0, 2, 1, 3)  # [B, T, F, 2]
        
        # Encoding
        encoded = self.encoder(encoder_input)
        
        # Decoding
        enhanced_mag = self.magnitude_decoder(encoded).squeeze(-1).permute(0, 2, 1)
        enhanced_phase = self.phase_decoder(encoded).squeeze(-1).permute(0, 2, 1)
        
        # Reconstruct complex spectrogram
        enhanced_spec = torch.polar(enhanced_mag, enhanced_phase)
        
        # iSTFT
        enhanced_wav = self.stft.inverse(enhanced_spec)
        
        # Обрезаем до исходной длины (или добавляем padding если нужно)
        if enhanced_wav.shape[-1] > original_length:
            enhanced_wav = enhanced_wav[..., :original_length]
        elif enhanced_wav.shape[-1] < original_length:
            pad_len = original_length - enhanced_wav.shape[-1]
            enhanced_wav = F.pad(enhanced_wav, (0, pad_len))
        
        return enhanced_wav, enhanced_mag, enhanced_phase, noisy_mag, noisy_phase

