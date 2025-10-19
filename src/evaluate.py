import torch
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm
import torchaudio
import pandas as pd
import collections

# Добавляем путь к src в sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.model import SpeechEnhancementModel
from src.data_loader import VoiceBankDemandDataset
from src.metrics import URGENTMetrics # Импортируем наш новый класс

def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Загрузка модели ---
    model = SpeechEnhancementModel(channels=args.channels, num_blocks=args.num_blocks).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # --- 2. Загрузка тестовых данных ---
    test_dataset = VoiceBankDemandDataset(data_dir=args.data_dir, is_train=False, segment_length_sec=None)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Словарь для хранения всех результатов
    all_metrics = collections.defaultdict(list)
    results_list = []

    # --- 3. Цикл оценки ---
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluating")
        for noisy_amp, noisy_pha, _, _, file_name in progress_bar:
            file_name = file_name[0]
            noisy_amp, noisy_pha = noisy_amp.to(device), noisy_pha.to(device)

            # --- Получение и сохранение улучшенного аудио ---
            _, _, pred_com = model(noisy_amp, noisy_pha)
            pred_com_pytorch = torch.complex(pred_com[..., 0], pred_com[..., 1])
            
            enhanced_wav = torch.istft(
                pred_com_pytorch.squeeze(0),
                n_fft=test_dataset.n_fft,
                hop_length=test_dataset.hop_length,
                win_length=test_dataset.win_length,
                window=test_dataset.window.to(device)
            ).cpu()

            output_path = os.path.join(args.output_dir, file_name)
            torchaudio.save(output_path, enhanced_wav.unsqueeze(0), test_dataset.sample_rate)

            # --- Расчет всех метрик ---
            clean_path = os.path.join(test_dataset.clean_dir, file_name)
            metrics = URGENTMetrics.compute_all_metrics(clean_path, output_path)
            
            # Сохраняем метрики
            metrics['file'] = file_name
            results_list.append(metrics)
            for key, value in metrics.items():
                if key != 'file':
                    all_metrics[key].append(value)
            
            progress_bar.set_postfix(pesq=metrics['PESQ'], si_sdr=metrics['SI-SDR'])

    # --- 4. Агрегация и вывод результатов ---
    avg_metrics = {key: sum(values)/len(values) for key, values in all_metrics.items()}
    
    print("\n--- Evaluation Finished ---")
    print("Average Metrics:")
    for key, value in avg_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Сохранение результатов в CSV файл
    df = pd.DataFrame(results_list)
    df.to_csv(os.path.join(args.output_dir, '_results.csv'), index=False)
    print(f"\nDetailed results saved to {os.path.join(args.output_dir, '_results.csv')}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained Speech Enhancement Model")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model (.pth file)")
    parser.add_argument('--data_dir', type=str, required=True, help="Path to the dataset directory")
    parser.add_argument('--output_dir', type=str, default='evaluation_results', help="Directory to save enhanced audio and scores")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of workers for dataloader")
    
    parser.add_argument('--channels', type=int, default=64, help="Number of channels in the model")
    parser.add_argument('--num_blocks', type=int, default=4, help="Number of processing blocks")
    
    args = parser.parse_args()
    
    # Исправляем проблему с импортом F.pad
    import torch.nn.functional as F
    from src import data_loader
    data_loader.F = F
    
    evaluate(args)