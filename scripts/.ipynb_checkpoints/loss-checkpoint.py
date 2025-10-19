import torch
import torch.nn as nn
import torch.nn.functional as F

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # вес для time-domain loss
        self.beta = beta  # вес для magnitude loss
        self.gamma = gamma  # вес для phase loss

    def forward(
        self, pred_wav, clean_wav, pred_mag, clean_mag, pred_phase, clean_phase
    ):
        # Time domain loss
        time_loss = F.l1_loss(pred_wav, clean_wav)

        # Magnitude loss
        mag_loss = F.mse_loss(pred_mag, clean_mag)

        # Phase loss
        phase_diff = torch.cos(pred_phase - clean_phase)
        phase_loss = 1 - phase_diff.mean()

        total_loss = (
            self.alpha * time_loss + self.beta * mag_loss + self.gamma * phase_loss
        )

        return total_loss, {
            "time_loss": time_loss.item(),
            "mag_loss": mag_loss.item(),
            "phase_loss": phase_loss.item(),
        }