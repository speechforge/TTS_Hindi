import os
import sys
import numpy as np
import librosa
import soundfile as sf
import torch
from pesq import pesq
from pymcd.mcd import Calculate_MCD
from frechet_audio_distance import FrechetAudioDistance
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import tempfile
import shutil
import warnings
warnings.filterwarnings("ignore")

def calculate_msd(ref_audio, deg_audio, sr):
    # Calculate log mel spectrograms
    ref_mel = librosa.power_to_db(librosa.feature.melspectrogram(y=ref_audio, sr=sr, n_mels=80))
    deg_mel = librosa.power_to_db(librosa.feature.melspectrogram(y=deg_audio, sr=sr, n_mels=80))
    
    # Align frames using DTW
    distance, path = fastdtw(ref_mel.T, deg_mel.T, dist=euclidean)
    
    # Calculate Mean Squared Distance along the warped path
    squared_diffs = []
    for x, y in path:
        diff = ref_mel[:, x] - deg_mel[:, y]
        squared_diffs.append(np.mean(diff ** 2))
        
    return np.mean(squared_diffs)

def evaluate(baseline_path, proposed_path):
    print("Loading and resampling audio to 16kHz for PESQ...")
    ref_16k, _ = librosa.load(baseline_path, sr=16000)
    deg_16k, _ = librosa.load(proposed_path, sr=16000)
    
    # Trim to same length for PESQ
    min_len = min(len(ref_16k), len(deg_16k))
    ref_16k_trimmed = ref_16k[:min_len]
    deg_16k_trimmed = deg_16k[:min_len]
    
    print("Calculating PESQ...")
    pesq_score = pesq(16000, ref_16k_trimmed, deg_16k_trimmed, 'wb')
    
    print("Calculating MCD...")
    mcd_toolbox = Calculate_MCD(MCD_mode="plain")
    mcd_score = mcd_toolbox.calculate_mcd(baseline_path, proposed_path)
    
    print("Calculating MSD...")
    msd_score = calculate_msd(ref_16k, deg_16k, 16000)
    
    print("Calculating UTMOS...")
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    
    baseline_utmos = predictor(torch.from_numpy(ref_16k).unsqueeze(0), 16000).item()
    proposed_utmos = predictor(torch.from_numpy(deg_16k).unsqueeze(0), 16000).item()
    
    print("Calculating FAD...")
    # Pre-cache vggish model with trust_repo=True to bypass interactive prompt
    torch.hub.load('harritaylor/torchvggish', 'vggish', trust_repo=True)
    
    # FAD requires directories
    with tempfile.TemporaryDirectory() as ref_dir, tempfile.TemporaryDirectory() as deg_dir:
        shutil.copy(baseline_path, os.path.join(ref_dir, "base.wav"))
        shutil.copy(proposed_path, os.path.join(deg_dir, "prop.wav"))
        fad = FrechetAudioDistance(
            model_name="vggish",
            use_pca=False, 
            use_activation=False,
            verbose=False
        )
        fad_score = fad.score(ref_dir, deg_dir, dtype="float32")
        
    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    print(f"PESQ (Wideband):  {pesq_score:.8f} (Higher is better)")
    print(f"MCD:              {mcd_score:.8f} (Lower is better)")
    print(f"MSD:              {msd_score:.8f} (Lower is better)")
    print(f"FAD:              {fad_score:.8f} (Lower is better)")
    print("-" * 40)
    print(f"UTMOS Baseline:   {baseline_utmos:.8f} (Higher is better)")
    print(f"UTMOS Proposed:   {proposed_utmos:.8f} (Higher is better)")
    print("="*40)

if __name__ == "__main__":
    baseline = "</path/to/your/baseline.wav>"
    proposed = "</path/to/your/proposed.wav>"
    evaluate(baseline, proposed)
