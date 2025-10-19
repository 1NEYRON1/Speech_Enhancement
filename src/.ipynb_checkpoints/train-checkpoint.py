import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import torch.nn.functional as F

# Добавляем путь к src в sys.path, чтобы можно было импортировать модули
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SpeechEnhancementModel
from src.data_loader import VoiceBankDemandDataset
from src.loss import CombinedLoss

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Загрузка данных ---
    train_dataset = VoiceBankDemandDataset(data_dir=args.data_dir, is_train=True)
    val_dataset = VoiceBankDemandDataset(data_dir=args.data_dir, is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    # --- 2. Инициализация модели, лосса и оптимизатора ---
    model = SpeechEnhancementModel(channels=args.channels, num_blocks=args.num_blocks).to(device)
    criterion = CombinedLoss(alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Создаем папку для сохранения моделей
    os.makedirs(args.save_dir, exist_ok=True)
    best_val_loss = float('inf')

    # --- 3. Цикл обучения ---
    for epoch in range(args.epochs):
        # -- Тренировка --
        model.train()
        train_loss_total = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        for noisy_amp, noisy_pha, clean_amp, clean_pha, _ in progress_bar:
            noisy_amp, noisy_pha = noisy_amp.to(device), noisy_pha.to(device)
            clean_amp, clean_pha = clean_amp.to(device), clean_pha.to(device)

            optimizer.zero_grad()
            
            pred_amp, pred_pha, _ = model(noisy_amp, noisy_pha)
            
            total_loss, amp_loss, ip_loss, gd_loss = criterion(pred_amp, pred_pha, clean_amp, clean_pha)
            
            total_loss.backward()
            optimizer.step()
            
            train_loss_total += total_loss.item()
            progress_bar.set_postfix(loss=total_loss.item(), amp=amp_loss.item(), phase=ip_loss.item())

        avg_train_loss = train_loss_total / len(train_loader)
        
        # -- Валидация --
        model.eval()
        val_loss_total = 0.0
        with torch.no_grad():
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            for noisy_amp, noisy_pha, clean_amp, clean_pha, _ in progress_bar_val:
                noisy_amp, noisy_pha = noisy_amp.to(device), noisy_pha.to(device)
                clean_amp, clean_pha = clean_amp.to(device), clean_pha.to(device)

                pred_amp, pred_pha, _ = model(noisy_amp, noisy_pha)
                
                total_loss, _, _, _ = criterion(pred_amp, pred_pha, clean_amp, clean_pha)
                val_loss_total += total_loss.item()
                progress_bar_val.set_postfix(loss=total_loss.item())

        avg_val_loss = val_loss_total / len(val_loader)

        print(f"Epoch {epoch+1}/{args.epochs} -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # -- Сохранение лучшей модели --
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pth'))
            print(f"Model saved! Best val loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Speech Enhancement Model")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--save_dir', type=str, default='saved_models', help="Directory to save trained models")
    parser.add_argument('--epochs', type=int, default=1, help="Number of training epochs")
    parser.add_argument('--batch_size', type=int, default=12, help="Batch size")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for dataloader")
    
    # Model parameters
    parser.add_argument('--channels', type=int, default=64, help="Number of channels in the model")
    parser.add_argument('--num_blocks', type=int, default=4, help="Number of processing blocks (Mamba+xLSTM)")
    
    # Loss parameters
    parser.add_argument('--alpha', type=float, default=0.5, help="Weight for amplitude loss")
    parser.add_argument('--beta', type=float, default=0.3, help="Weight for instantaneous phase loss")
    parser.add_argument('--gamma', type=float, default=0.2, help="Weight for group delay loss")
    
    args = parser.parse_args()
    
    train(args)