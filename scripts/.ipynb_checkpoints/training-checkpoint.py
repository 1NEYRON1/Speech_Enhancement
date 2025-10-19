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
from .utils.create_dataset import VoiceBankDataset
from .utils.compute_metrics import URGENTMetrics
from .models import *
from .loss import CombinedLoss

def validate(model, val_loader, criterion, device, config, save_audio=False):
    model.eval()
    total_loss = 0
    metrics = {'PESQ': [], 'STOI': [], 'eSTOI': [], 'SI-SDR': []}
    
    enhanced_dir = Path(config.results_path) / "enhanced_audio"
    if save_audio:
        enhanced_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            noisy = batch['noisy'].to(device)
            clean = batch['clean'].to(device)
            filename = batch['filename'][0]
            clean_path = batch['clean_path'][0]
            
            enhanced_wav, pred_mag, pred_phase, _, _ = model(noisy)
            
            clean_spec = model.stft(clean)
            clean_mag = torch.abs(clean_spec)
            clean_phase = torch.angle(clean_spec)
            
            loss, _ = criterion(
                enhanced_wav, clean,
                pred_mag, clean_mag,
                pred_phase, clean_phase
            )
            
            total_loss += loss.item()
            
            # Сохраняем enhanced audio
            if save_audio:
                enhanced_path = enhanced_dir / filename
            else:
                enhanced_path = "/tmp/" + filename
            
            torchaudio.save(
                str(enhanced_path),
                enhanced_wav[0].cpu().unsqueeze(0),
                config.sample_rate
            )

            batch_metrics = URGENTMetrics.compute_all_metrics(
                clean_path,
                str(enhanced_path),
                sr=config.sample_rate
            )
            
            for key, value in batch_metrics.items():
                metrics[key].append(value)
            
            # Удаляем временный файл
            if not save_audio and Path(enhanced_path).exists():
                Path(enhanced_path).unlink()
            
            pbar.set_postfix({'PESQ': f"{np.mean(metrics['PESQ']):.3f}"})
    
    avg_loss = total_loss / len(val_loader)
    avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
    
    return avg_loss, avg_metrics

def train_epoch(model, train_loader, optimizer, criterion, device, config):
    model.train()
    total_loss = 0
    loss_components = {'time_loss': 0, 'mag_loss': 0, 'phase_loss': 0}
    
    pbar = tqdm(train_loader, desc="Training")
    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        
        optimizer.zero_grad()
        enhanced_wav, pred_mag, pred_phase, _, _ = model(noisy)
        
        clean_spec = model.stft(clean)
        clean_mag = torch.abs(clean_spec)
        clean_phase = torch.angle(clean_spec)
        
        loss, loss_dict = criterion(
            enhanced_wav, clean, 
            pred_mag, clean_mag, 
            pred_phase, clean_phase
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
        optimizer.step()
        
        total_loss += loss.item()
        for key in loss_components:
            loss_components[key] += loss_dict[key]
        
        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    avg_loss = total_loss / len(train_loader)
    avg_components = {k: v / len(train_loader) for k, v in loss_components.items()}
    
    return avg_loss, avg_components

def train_model(config, device, continue_training_best=True):
    Path(config.checkpoint_path).mkdir(parents=True, exist_ok=True)
    Path(config.results_path).mkdir(parents=True, exist_ok=True)
    
    print("Loading datasets...")
    train_dataset = VoiceBankDataset(
        config.train_clean_path,
        config.train_noisy_path,
        mode='train'
    )
    test_dataset = VoiceBankDataset(
        config.test_clean_path,
        config.test_noisy_path,
        max_len=None,
        mode='test'
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2
    )
    
    print("\nInitializing MP-SENet model...")
    model = MPSENet(config).to(device)
    if continue_training_best:
        checkpoint = torch.load(f"{config.checkpoint_path}/best_model.pt", map_location='cuda', weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params / 1e6:.2f}M")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    criterion = CombinedLoss()
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_pesq': [],
        'val_stoi': [],
        'val_estoi': [],
        'val_sisdr': []
    }
    
    best_pesq = 0.0
    
    print("\nStarting training...")
    print("="*70)
    
    for epoch in range(config.num_epochs):
        print(f"\nEpoch {epoch+1}/{config.num_epochs}")
        print("-" * 70)

        train_loss, train_components = train_epoch(
            model, train_loader, optimizer, criterion, device, config
        )
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Time: {train_components['time_loss']:.4f}, "
              f"Mag: {train_components['mag_loss']:.4f}, "
              f"Phase: {train_components['phase_loss']:.4f}")

        val_loss, val_metrics = validate(model, test_loader, criterion, device, config, save_audio=False)
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Metrics:")
        print(f"PESQ:   {val_metrics['PESQ']:.4f}")
        print(f"STOI:   {val_metrics['STOI']:.4f}")
        print(f"eSTOI:  {val_metrics['eSTOI']:.4f}")
        print(f"SI-SDR: {val_metrics['SI-SDR']:.2f} dB")

        scheduler.step(val_metrics['PESQ'])

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_pesq'].append(val_metrics['PESQ'])
        history['val_stoi'].append(val_metrics['STOI'])
        history['val_estoi'].append(val_metrics['eSTOI'])
        history['val_sisdr'].append(val_metrics['SI-SDR'])

        if val_metrics['PESQ'] > best_pesq:
            best_pesq = val_metrics['PESQ']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'pesq': best_pesq,
                'metrics': val_metrics,
                'config': config
            }, f"{config.checkpoint_path}/best_model.pt")
            print(f"Saved best model with PESQ: {best_pesq:.4f}")
            
            if best_pesq >= 3.25:
                print(f"Target PESQ {trainer_ner} achieved!")
    
    pd.DataFrame(history).to_csv(f"{config.results_path}/training_history.csv", index=False)
    print("\n" + "="*70)
    print("Training completed!")
    
    return model, history