import os
import sys
import json
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

import commons
import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
from scipy.io.wavfile import write

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def main():
    if len(sys.argv) < 3:
        print("Usage: python infer.py <checkpoint_path> <output_dir>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    output_dir = sys.argv[2]

    hps = utils.get_hparams_from_file("./experiments/run_20260629_162133_hindi_phase3_mamba/config.json")
    
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cpu()
    _ = net_g.eval()

    _ = utils.load_checkpoint(checkpoint_path, net_g, None)

    # Read the first 10 sentences from the validation list
    sentences = []
    with open("others/filelists/hindi_val.txt.cleaned", "r", encoding="utf-8") as f:
        for _ in range(10):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split('|')
            if len(parts) >= 2:
                sentences.append(parts[1])
            else:
                sentences.append(parts[0])

    os.makedirs(output_dir, exist_ok=True)

    for i, text in enumerate(sentences):
        print(f"Generating sentence {i+1}: {text}")
        stn_tst = get_text(text, hps)
        with torch.no_grad():
            x_tst = stn_tst.cpu().unsqueeze(0)
            x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
            audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][0,0].data.cpu().float().numpy()
            
            write(os.path.join(output_dir, f"sentence_{i+1}.wav"), hps.data.sampling_rate, audio)
    
    print(f"TTS generation complete. Files saved in '{output_dir}' directory.")

if __name__ == '__main__':
    main()
