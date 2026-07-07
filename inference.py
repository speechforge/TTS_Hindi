import os
import torch
import torchaudio
import argparse
import scipy.io.wavfile as wavfile
from transformers import WavLMModel, AutoFeatureExtractor

import utils
from models import SynthesizerTrn
from text.symbols import symbols
from text import text_to_sequence
import commons

def extract_wavlm_embedding(wav_path, device):
    print(f"Extracting WavLM embedding from {wav_path}...")
    model_name = "microsoft/wavlm-base-plus"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    model = WavLMModel.from_pretrained(model_name).to(device)
    model.eval()

    target_sr = feature_extractor.sampling_rate
    wav, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    with torch.no_grad():
        inputs = feature_extractor(wav.squeeze(0).numpy(), sampling_rate=target_sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        frame_emb = last_hidden_state.squeeze(0) # [time, 768]

    return frame_emb

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize in Hindi")
    parser.add_argument("--ref_audio", type=str, required=True, help="Path to reference audio for voice cloning")
    parser.add_argument("--output", type=str, default="output.wav", help="Path to output synthesized audio")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to generator checkpoint (e.g., G_last.pth)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.json")
    parser.add_argument("--sid", type=int, default=0, help="Speaker ID (0 or 1)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    hps = utils.get_hparams_from_file(args.config)
    
    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model).to(device)
    
    net_g.eval()
    
    print(f"Loading checkpoint {args.checkpoint}...")
    utils.load_checkpoint(args.checkpoint, net_g, None)
    
    wavlm_emb = extract_wavlm_embedding(args.ref_audio, device)
    wavlm_emb = wavlm_emb.unsqueeze(0) # [1, Time, 768]
    wavlm_lengths = torch.LongTensor([wavlm_emb.size(1)]).to(device)

    stn_tst = get_text(args.text, hps)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(device)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)
        sid = torch.LongTensor([args.sid]).to(device)

        print("Synthesizing audio...")
        audio, _, _, *_ = net_g.infer(
            x_tst, 
            x_tst_lengths, 
            sid=sid,
            noise_scale=.667, 
            noise_scale_w=0.8, 
            length_scale=1,
            wavlm_emb=wavlm_emb,
            wavlm_lengths=wavlm_lengths
        )
        audio = audio[0][0].data.cpu().float().numpy()

    print(f"Saving audio to {args.output}...")
    wavfile.write(args.output, hps.data.sampling_rate, audio)
    print("Done!")

if __name__ == "__main__":
    main()
