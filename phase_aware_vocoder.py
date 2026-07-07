import math

import torch
from torch import nn
from torch.nn import functional as F


def _as_btc(f0):
  if f0.dim() == 2:
    return f0.unsqueeze(-1)
  if f0.dim() == 3 and f0.size(1) == 1:
    return f0.transpose(1, 2)
  if f0.dim() == 3:
    return f0
  raise ValueError("Expected f0 with shape [B, T], [B, 1, T], or [B, T, 1].")


def smooth_f0_voiced(f0, voiced_mask, kernel_size=5):
  if kernel_size <= 1:
    return f0
  if kernel_size % 2 == 0:
    raise ValueError("kernel_size must be odd.")
  f0 = _as_btc(f0).transpose(1, 2)
  voiced_mask = _as_btc(voiced_mask).transpose(1, 2).to(dtype=f0.dtype)
  weight = torch.ones(1, 1, kernel_size, dtype=f0.dtype, device=f0.device)
  pad = kernel_size // 2
  numer = F.conv1d(f0 * voiced_mask, weight, padding=pad)
  denom = F.conv1d(voiced_mask, weight, padding=pad).clamp_min(1.0)
  return (numer / denom).transpose(1, 2)


def voiced_region_mask(f0, voiced_threshold=10.0, smooth_frames=3):
  f0 = _as_btc(f0)
  mask = (f0 > voiced_threshold).to(dtype=f0.dtype)
  if smooth_frames > 1:
    if smooth_frames % 2 == 0:
      smooth_frames += 1
    weight = torch.ones(1, 1, smooth_frames, dtype=f0.dtype, device=f0.device)
    smoothed = F.conv1d(mask.transpose(1, 2), weight, padding=smooth_frames // 2)
    mask = (smoothed >= (smooth_frames // 2 + 1)).to(dtype=f0.dtype).transpose(1, 2)
  return mask


def continuous_phase_from_f0(f0, sampling_rate, hop_length, voiced_threshold=10.0, smooth_frames=5):
  f0 = _as_btc(f0)
  voiced = voiced_region_mask(f0, voiced_threshold=voiced_threshold, smooth_frames=smooth_frames)
  f0_smooth = smooth_f0_voiced(f0, voiced, kernel_size=smooth_frames)
  radians_per_frame = 2.0 * math.pi * f0_smooth * float(hop_length) / float(sampling_rate)
  phase = torch.cumsum(radians_per_frame * voiced, dim=1)
  return phase, voiced


class PhaseAwareHarmonicSource(nn.Module):
  """Continuous-phase harmonic excitation for voiced regions.

  The module mirrors the source-generation idea used by iSTFT/NSF vocoders:
  preserve accumulated phase through voiced spans, suppress harmonic excitation in
  unvoiced regions, and keep a separate noise branch for unvoiced/detail energy.
  """

  def __init__(self,
      sampling_rate,
      hop_length,
      harmonic_num=8,
      sine_amp=0.1,
      noise_std=0.003,
      voiced_threshold=10.0,
      smooth_frames=5):
    super().__init__()
    self.sampling_rate = sampling_rate
    self.hop_length = hop_length
    self.harmonic_num = harmonic_num
    self.sine_amp = sine_amp
    self.noise_std = noise_std
    self.voiced_threshold = voiced_threshold
    self.smooth_frames = smooth_frames
    self.merge = nn.Linear(harmonic_num + 1, 1)

  def forward(self, f0):
    f0 = _as_btc(f0)
    base_phase, voiced = continuous_phase_from_f0(
      f0,
      sampling_rate=self.sampling_rate,
      hop_length=self.hop_length,
      voiced_threshold=self.voiced_threshold,
      smooth_frames=self.smooth_frames)
    harmonics = torch.arange(
      1,
      self.harmonic_num + 2,
      dtype=f0.dtype,
      device=f0.device).view(1, 1, -1)
    sine = torch.sin(base_phase * harmonics) * self.sine_amp
    sine = sine * voiced

    voiced_noise = torch.randn_like(sine) * self.noise_std
    unvoiced_noise = torch.randn_like(sine) * (self.sine_amp / 3.0)
    noise = voiced_noise * voiced + unvoiced_noise * (1.0 - voiced)
    harmonic_source = self.merge(sine + noise)
    return harmonic_source, noise, voiced


class PhaseAwarePitchShift:
  """Utility for smooth dynamic pitch curves before vocoder conditioning."""

  def __init__(self, min_f0=20.0, max_f0=1200.0, smoothing_frames=5):
    self.min_f0 = min_f0
    self.max_f0 = max_f0
    self.smoothing_frames = smoothing_frames

  def __call__(self, f0, semitone_shift):
    f0 = _as_btc(f0)
    if not torch.is_tensor(semitone_shift):
      semitone_shift = torch.tensor(semitone_shift, dtype=f0.dtype, device=f0.device)
    ratio = torch.pow(torch.tensor(2.0, dtype=f0.dtype, device=f0.device), semitone_shift / 12.0)
    while ratio.dim() < f0.dim():
      ratio = ratio.unsqueeze(-1)
    shifted = torch.clamp(f0 * ratio, self.min_f0, self.max_f0)
    voiced = voiced_region_mask(f0, voiced_threshold=self.min_f0, smooth_frames=self.smoothing_frames)
    shifted = smooth_f0_voiced(shifted, voiced, kernel_size=self.smoothing_frames)
    return shifted * voiced
