import torch
import torch.nn as nn
import numpy as np

def anti_wrapping_function(x):
    """
    Функция для "развертывания" фазы, чтобы избежать скачков на 2*pi.
    Перенесено из MP-SENet.
    """
    return torch.abs(x - torch.round(x / (2 * np.pi)) * 2 * np.pi)

class CombinedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.2, gamma=0.3):
        """
        Комбинированная функция потерь.
        
        Args:
            alpha (float): Вес для amplitude loss (L1).
            beta (float): Вес для instantaneous phase loss.
            gamma (float): Вес для group delay loss.
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()

    def forward(self, pred_amp, pred_pha, true_amp, true_pha):
        # 1. Потери на амплитуде
        amp_loss = self.l1_loss(pred_amp, true_amp)
        
        # 2. Потери на фазе (Instantaneous Phase)
        ip_loss = torch.mean(anti_wrapping_function(pred_pha - true_pha))
        
        # 3. Потери на групповой задержке (Group Delay)
        # Разница по оси времени
        pred_gd = torch.diff(pred_pha, dim=-1)
        true_gd = torch.diff(true_pha, dim=-1)
        gd_loss = torch.mean(anti_wrapping_function(pred_gd - true_gd))
        
        # Комбинируем потери
        total_loss = self.alpha * amp_loss + self.beta * ip_loss + self.gamma * gd_loss
        
        return total_loss, amp_loss, ip_loss, gd_loss