import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm


class LinearNorm(nn.Module):
  def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
    super().__init__()
    self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.xavier_uniform_(
      self.linear_layer.weight,
      gain=nn.init.calculate_gain(w_init_gain))

  def forward(self, x):
    return self.linear_layer(x)


class AdaLayerNorm(nn.Module):
  def __init__(self, style_dim, channels, eps=1e-5):
    super().__init__()
    self.channels = channels
    self.eps = eps
    self.fc = nn.Linear(style_dim, channels * 2)

  def forward(self, x, style):
    # x: [B, C, T], style: [B, S]
    h = self.fc(style).unsqueeze(-1)
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    x = F.layer_norm(
      x.transpose(1, 2),
      (self.channels,),
      eps=self.eps).transpose(1, 2)
    return (1 + gamma) * x + beta


class AdaIN1d(nn.Module):
  def __init__(self, style_dim, channels):
    super().__init__()
    self.norm = nn.InstanceNorm1d(channels, affine=False)
    self.fc = nn.Linear(style_dim, channels * 2)

  def forward(self, x, style):
    h = self.fc(style).unsqueeze(-1)
    gamma, beta = torch.chunk(h, chunks=2, dim=1)
    return (1 + gamma) * self.norm(x) + beta


class UpSample1d(nn.Module):
  def __init__(self, layer_type):
    super().__init__()
    self.layer_type = layer_type

  def forward(self, x):
    if self.layer_type == "none":
      return x
    return F.interpolate(x, scale_factor=2, mode="nearest")


class AdainResBlk1d(nn.Module):
  def __init__(self, dim_in, dim_out, style_dim, upsample="none", dropout_p=0.0):
    super().__init__()
    self.actv = nn.LeakyReLU(0.2)
    self.upsample = UpSample1d(upsample)
    self.learned_sc = dim_in != dim_out
    self.dropout = nn.Dropout(dropout_p)
    self.norm1 = AdaIN1d(style_dim, dim_in)
    self.norm2 = AdaIN1d(style_dim, dim_out)
    self.conv1 = weight_norm(nn.Conv1d(dim_in, dim_out, 3, 1, 1))
    self.conv2 = weight_norm(nn.Conv1d(dim_out, dim_out, 3, 1, 1))
    if upsample == "none":
      self.pool = nn.Identity()
    else:
      self.pool = weight_norm(nn.ConvTranspose1d(
        dim_in, dim_in, kernel_size=3, stride=2, groups=dim_in,
        padding=1, output_padding=1))
    if self.learned_sc:
      self.conv1x1 = weight_norm(nn.Conv1d(dim_in, dim_out, 1, 1, 0, bias=False))

  def _shortcut(self, x):
    x = self.upsample(x)
    if self.learned_sc:
      x = self.conv1x1(x)
    return x

  def _residual(self, x, style):
    x = self.norm1(x, style)
    x = self.actv(x)
    x = self.pool(x)
    x = self.conv1(self.dropout(x))
    x = self.norm2(x, style)
    x = self.actv(x)
    x = self.conv2(self.dropout(x))
    return x

  def forward(self, x, style):
    return (self._residual(x, style) + self._shortcut(x)) / math.sqrt(2)


class StyleTTS2DurationEncoder(nn.Module):
  def __init__(self, style_dim, hidden_channels, n_layers, dropout=0.1):
    super().__init__()
    self.hidden_channels = hidden_channels
    self.style_dim = style_dim
    self.dropout = dropout
    self.lstm_layers = nn.ModuleList()
    self.norm_layers = nn.ModuleList()
    for _ in range(n_layers):
      self.lstm_layers.append(nn.LSTM(
        hidden_channels + style_dim,
        hidden_channels // 2,
        num_layers=1,
        batch_first=True,
        bidirectional=True))
      self.norm_layers.append(AdaLayerNorm(style_dim, hidden_channels))

  def forward(self, x, style, text_lengths, x_mask):
    # x: [B, C, T], x_mask: [B, 1, T]
    h = x * x_mask
    lengths = text_lengths.detach().cpu()
    for lstm, norm in zip(self.lstm_layers, self.norm_layers):
      style_seq = style.unsqueeze(-1).expand(-1, -1, h.size(2))
      h_in = torch.cat([h, style_seq], dim=1).transpose(1, 2)
      packed = nn.utils.rnn.pack_padded_sequence(
        h_in, lengths, batch_first=True, enforce_sorted=False)
      lstm.flatten_parameters()
      packed, _ = lstm(packed)
      h_out, _ = nn.utils.rnn.pad_packed_sequence(
        packed, batch_first=True, total_length=x_mask.size(2))
      h = h_out.transpose(1, 2)
      h = F.dropout(h, p=self.dropout, training=self.training)
      h = norm(h, style) * x_mask
    return h


class StyleTTS2Predictors(nn.Module):
  """StyleTTS2-style duration, F0 and energy heads adapted to VITS encoder states."""

  def __init__(self,
      hidden_channels,
      style_dim=128,
      n_layers=2,
      max_duration=512,
      dropout=0.1,
      prosody_upsample=False):
    super().__init__()
    if hidden_channels % 2 != 0:
      raise ValueError("StyleTTS2Predictors requires an even hidden channel count.")
    self.hidden_channels = hidden_channels
    self.style_dim = style_dim
    self.max_duration = max_duration

    self.duration_encoder = StyleTTS2DurationEncoder(
      style_dim=style_dim,
      hidden_channels=hidden_channels,
      n_layers=n_layers,
      dropout=dropout)
    self.duration_lstm = nn.LSTM(
      hidden_channels + style_dim,
      hidden_channels // 2,
      num_layers=1,
      batch_first=True,
      bidirectional=True)
    self.duration_proj = LinearNorm(hidden_channels, max_duration)
    self.duration_reg_proj = nn.Conv1d(hidden_channels, 1, 1)

    self.shared_lstm = nn.LSTM(
      hidden_channels + style_dim,
      hidden_channels // 2,
      num_layers=1,
      batch_first=True,
      bidirectional=True)
    mid_channels = hidden_channels // 2
    upsample_mode = "nearest" if prosody_upsample else "none"
    self.f0 = nn.ModuleList([
      AdainResBlk1d(hidden_channels, hidden_channels, style_dim, dropout_p=dropout),
      AdainResBlk1d(hidden_channels, mid_channels, style_dim, upsample=upsample_mode, dropout_p=dropout),
      AdainResBlk1d(mid_channels, mid_channels, style_dim, dropout_p=dropout)])
    self.energy = nn.ModuleList([
      AdainResBlk1d(hidden_channels, hidden_channels, style_dim, dropout_p=dropout),
      AdainResBlk1d(hidden_channels, mid_channels, style_dim, upsample=upsample_mode, dropout_p=dropout),
      AdainResBlk1d(mid_channels, mid_channels, style_dim, dropout_p=dropout)])
    self.f0_proj = nn.Conv1d(mid_channels, 1, 1)
    self.energy_proj = nn.Conv1d(mid_channels, 1, 1)

  def _run_lstm(self, lstm, x, style, lengths, total_length):
    style_seq = style.unsqueeze(-1).expand(-1, -1, x.size(2))
    x = torch.cat([x, style_seq], dim=1).transpose(1, 2)
    packed = nn.utils.rnn.pack_padded_sequence(
      x, lengths.detach().cpu(), batch_first=True, enforce_sorted=False)
    lstm.flatten_parameters()
    packed, _ = lstm(packed)
    x, _ = nn.utils.rnn.pad_packed_sequence(
      packed, batch_first=True, total_length=total_length)
    return x.transpose(1, 2)

  def _duration_expectation(self, duration_logits):
    bins = torch.arange(
      self.max_duration,
      device=duration_logits.device,
      dtype=duration_logits.dtype)
    return torch.sum(torch.softmax(duration_logits, dim=-1) * bins, dim=-1)

  def _apply_alignment(self, x, alignment):
    if alignment is None:
      return x
    if alignment.dim() == 4:
      alignment = alignment.squeeze(1)
    return torch.matmul(x, alignment.transpose(1, 2))

  def forward(self, x, x_lengths, x_mask, style, alignment=None, frame_lengths=None):
    duration_hidden = self.duration_encoder(x, style, x_lengths, x_mask)
    duration_lstm = self._run_lstm(
      self.duration_lstm,
      duration_hidden,
      style,
      x_lengths,
      total_length=x_mask.size(2))
    duration_logits = self.duration_proj(duration_lstm.transpose(1, 2))
    duration_logits = duration_logits.masked_fill(x_mask.transpose(1, 2) == 0, -1e4)
    log_duration = self.duration_reg_proj(duration_lstm) * x_mask

    prosody_input = self._apply_alignment(duration_hidden, alignment)
    if frame_lengths is None:
      if alignment is None:
        frame_lengths = x_lengths
      else:
        alignment_for_lengths = alignment.squeeze(1) if alignment.dim() == 4 else alignment
        frame_lengths = torch.clamp_min(alignment_for_lengths.sum(dim=-1).sum(dim=-1).long(), 1)
    prosody_mask = torch.ones(
      prosody_input.size(0), 1, prosody_input.size(2),
      dtype=prosody_input.dtype,
      device=prosody_input.device)
    max_len = prosody_input.size(2)
    if frame_lengths is not None:
      frame_mask = torch.arange(max_len, device=prosody_input.device).unsqueeze(0) < frame_lengths.to(prosody_input.device).unsqueeze(1)
      prosody_mask = frame_mask.unsqueeze(1).to(prosody_input.dtype)

    shared = self._run_lstm(
      self.shared_lstm,
      prosody_input * prosody_mask,
      style,
      frame_lengths,
      total_length=max_len)
    f0 = shared
    for block in self.f0:
      f0 = block(f0, style)
    energy = shared
    for block in self.energy:
      energy = block(energy, style)

    return {
      "duration_logits": duration_logits,
      "duration": self._duration_expectation(duration_logits) * x_mask.squeeze(1),
      "log_duration": log_duration,
      "f0": self.f0_proj(f0).squeeze(1),
      "energy": self.energy_proj(energy).squeeze(1),
      "prosody_mask": prosody_mask.squeeze(1)
    }
