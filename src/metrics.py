import soundfile as sf
import numpy as np
from pesq import pesq
from pystoi import stoi

class URGENTMetrics:
    """
    Метрики из URGENT Challenge 2026
    Репозиторий: https://github.com/urgent-challenge/urgent2026_challenge_track1
    """

    @staticmethod
    def compute_pesq(clean_path, enhanced_path, sr=16000):
        """PESQ - Perceptual Evaluation of Speech Quality"""
        try:
            clean, _ = sf.read(clean_path)
            enhanced, _ = sf.read(enhanced_path)
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]
            score = pesq(sr, clean, enhanced, "wb")
            return score
        except Exception as e:
            print(f"Error computing PESQ for {clean_path}: {e}")
            return 0.0

    @staticmethod
    def compute_stoi(clean_path, enhanced_path, sr=16000):
        """STOI - Short-Time Objective Intelligibility"""
        try:
            clean, _ = sf.read(clean_path)
            enhanced, _ = sf.read(enhanced_path)
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]
            score = stoi(clean, enhanced, sr, extended=False)
            return score
        except Exception as e:
            print(f"Error computing STOI for {clean_path}: {e}")
            return 0.0

    @staticmethod
    def compute_estoi(clean_path, enhanced_path, sr=16000):
        """Extended STOI"""
        try:
            clean, _ = sf.read(clean_path)
            enhanced, _ = sf.read(enhanced_path)
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]
            score = stoi(clean, enhanced, sr, extended=True)
            return score
        except Exception as e:
            print(f"Error computing eSTOI for {clean_path}: {e}")
            return 0.0

    @staticmethod
    def compute_si_sdr(clean_path, enhanced_path):
        """SI-SDR - Scale-Invariant Signal-to-Distortion Ratio"""
        try:
            clean, _ = sf.read(clean_path)
            enhanced, _ = sf.read(enhanced_path)
            min_len = min(len(clean), len(enhanced))
            clean = clean[:min_len]
            enhanced = enhanced[:min_len]
            
            if np.sum(clean**2) == 0: return 0.0 # Избегаем деления на ноль
            
            alpha = np.dot(enhanced, clean) / np.dot(clean, clean)
            s_target = alpha * clean
            e_noise = enhanced - s_target
            
            if np.sum(e_noise**2) == 0: return float('inf') # Идеальное восстановление

            si_sdr = 10 * np.log10(np.sum(s_target**2) / np.sum(e_noise**2))
            return si_sdr
        except Exception as e:
            print(f"Error computing SI-SDR for {clean_path}: {e}")
            return 0.0

    @staticmethod
    def compute_all_metrics(clean_path, enhanced_path, sr=16000):
        """Вычисляет все метрики URGENT Challenge"""
        return {
            "PESQ": URGENTMetrics.compute_pesq(clean_path, enhanced_path, sr),
            "STOI": URGENTMetrics.compute_stoi(clean_path, enhanced_path, sr),
            "eSTOI": URGENTMetrics.compute_estoi(clean_path, enhanced_path, sr),
            "SI-SDR": URGENTMetrics.compute_si_sdr(clean_path, enhanced_path),
        }