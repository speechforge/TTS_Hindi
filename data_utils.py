import time
import os
import random
import logging
import numpy as np
import torch
import torch.utils.data

import commons 
from mel_processing import spectrogram_torch
from utils import load_wav_to_torch, load_filepaths_and_text
from text import text_to_sequence, cleaned_text_to_sequence

logger = logging.getLogger(__name__)


def _get_hparam(hparams, name, default=None):
    return getattr(hparams, name, default)


def _bucket_length(spec_length, text_length, bucket_by="spec", text_length_weight=6):
    if bucket_by == "text":
        return text_length
    if bucket_by in ("max", "balanced"):
        return max(spec_length, text_length * text_length_weight)
    if bucket_by == "sum":
        return spec_length + text_length * text_length_weight
    return spec_length


class TextAudioLoader(torch.utils.data.Dataset):
    """
        1) loads audio, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_and_text, hparams):
        self.audiopaths_and_text = load_filepaths_and_text(audiopaths_and_text)
        self.text_cleaners  = hparams.text_cleaners
        self.max_wav_value  = hparams.max_wav_value
        self.sampling_rate  = hparams.sampling_rate
        self.filter_length  = hparams.filter_length 
        self.hop_length     = hparams.hop_length 
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate 

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        self.cache_text = _get_hparam(hparams, "cache_text", True)
        self.bucket_by = _get_hparam(hparams, "bucket_by", "spec")
        self.text_length_weight = _get_hparam(hparams, "text_length_weight", 6)
        self.text_cache = []

        random.seed(1234)
        random.shuffle(self.audiopaths_and_text)
        self._filter()


    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_and_text_new = []
        lengths = []
        text_cache = []
        for audiopath, text in self.audiopaths_and_text:
            try:
                text_norm = self._text_to_tensor(text)
            except Exception as exc:
                logger.warning("Skipping sample with invalid text tokens: %s | %s", text[:80], exc)
                continue
            text_len = text_norm.size(0)
            if self.min_text_len <= text_len and text_len <= self.max_text_len:
                if not os.path.exists(audiopath):
                    logger.warning("Skipping missing audio file: %s", audiopath)
                    continue
                spec_length = os.path.getsize(audiopath) // (2 * self.hop_length)
                audiopaths_and_text_new.append([audiopath, text])
                text_cache.append(text_norm)
                lengths.append(_bucket_length(
                    spec_length,
                    text_len,
                    bucket_by=self.bucket_by,
                    text_length_weight=self.text_length_weight))
        self.audiopaths_and_text = audiopaths_and_text_new
        self.lengths = lengths
        self.text_cache = text_cache if self.cache_text else []
        if len(self.audiopaths_and_text) == 0:
            raise RuntimeError("No valid audio/text pairs remain after filtering.")

    def get_audio_text_pair(self, audiopath_and_text, index=None):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text, index=index)
        spec, wav = self.get_audio(audiopath)
        return (text, spec, wav)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} {} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def _text_to_tensor(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return torch.LongTensor(text_norm)

    def get_text(self, text, index=None):
        if self.cache_text and index is not None and index < len(self.text_cache):
            return self.text_cache[index].clone()
        return self._text_to_tensor(text)

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audiopaths_and_text[index], index=index)

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextAudioCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text and aduio
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audiopaths_sid_text, hparams):
        self.audiopaths_sid_text = load_filepaths_and_text(audiopaths_sid_text)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.filter_length  = hparams.filter_length
        self.hop_length     = hparams.hop_length
        self.win_length     = hparams.win_length
        self.sampling_rate  = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)
        self.cache_text = _get_hparam(hparams, "cache_text", True)
        self.bucket_by = _get_hparam(hparams, "bucket_by", "spec")
        self.text_length_weight = _get_hparam(hparams, "text_length_weight", 6)
        self.text_cache = []

        random.seed(1234)
        random.shuffle(self.audiopaths_sid_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """
        # Store spectrogram lengths for Bucketing
        # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
        # spec_length = wav_length // hop_length

        audiopaths_sid_text_new = []
        lengths = []
        text_cache = []
        for audiopath, sid, style_id, text in self.audiopaths_sid_text:
            try:
                text_norm = self._text_to_tensor(text)
            except Exception as exc:
                logger.warning("Skipping sample with invalid text tokens: %s | %s", text[:80], exc)
                continue
            text_len = text_norm.size(0)
            if self.min_text_len <= text_len and text_len <= self.max_text_len:
                if not os.path.exists(audiopath):
                    logger.warning("Skipping missing audio file: %s", audiopath)
                    continue
                spec_length = os.path.getsize(audiopath) // (2 * self.hop_length)
                audiopaths_sid_text_new.append([audiopath, sid, style_id, text])
                text_cache.append(text_norm)
                lengths.append(_bucket_length(
                    spec_length,
                    text_len,
                    bucket_by=self.bucket_by,
                    text_length_weight=self.text_length_weight))
        self.audiopaths_sid_text = audiopaths_sid_text_new
        self.lengths = lengths
        self.text_cache = text_cache if self.cache_text else []
        if len(self.audiopaths_sid_text) == 0:
            raise RuntimeError("No valid audio/speaker/text triples remain after filtering.")

    def get_audio_text_speaker_pair(self, audiopath_sid_text, index=None):
        # separate filename, speaker_id, style_id and text
        audiopath, sid, style_id, text = audiopath_sid_text[0], audiopath_sid_text[1], audiopath_sid_text[2], audiopath_sid_text[3]
        text = self.get_text(text, index=index)
        spec, wav = self.get_audio(audiopath)
        sid = self.get_sid(sid)
        
        wavlm_path = audiopath.replace(".wav", "_wavlm.pt")
        if os.path.exists(wavlm_path):
            wavlm_emb = torch.load(wavlm_path)
            if wavlm_emb.dim() == 1:
                wavlm_emb = wavlm_emb.unsqueeze(0)
        else:
            # Fallback
            wavlm_emb = torch.zeros(1, 768, dtype=torch.float32)

        return (text, spec, wav, sid, wavlm_emb)

    def get_audio(self, filename):
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.max_wav_value
        audio_norm = audio_norm.unsqueeze(0)
        spec_filename = filename.replace(".wav", ".spec.pt")
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename)
        else:
            spec = spectrogram_torch(audio_norm, self.filter_length,
                self.sampling_rate, self.hop_length, self.win_length,
                center=False)
            spec = torch.squeeze(spec, 0)
            torch.save(spec, spec_filename)
        return spec, audio_norm

    def _text_to_tensor(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        return torch.LongTensor(text_norm)

    def get_text(self, text, index=None):
        if self.cache_text and index is not None and index < len(self.text_cache):
            return self.text_cache[index].clone()
        return self._text_to_tensor(text)

    def get_sid(self, sid):
        sid = torch.LongTensor([int(sid)])
        return sid

    def __getitem__(self, index):
        return self.get_audio_text_speaker_pair(self.audiopaths_sid_text[index], index=index)

    def __len__(self):
        return len(self.audiopaths_sid_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: [text_normalized, spec_normalized, wav_normalized, sid, wavlm_emb]
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])
        max_wavlm_len = max([x[4].size(0) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))
        wavlm_lengths = torch.LongTensor(len(batch))
        sids = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        wavlm_padded = torch.FloatTensor(len(batch), 768, max_wavlm_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        wavlm_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)
            
            sids[i] = row[3][0]

            wavlm = row[4].transpose(0, 1) # [768, Time]
            wavlm_padded[i, :, :wavlm.size(1)] = wavlm
            wavlm_lengths[i] = wavlm.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sids, wavlm_padded, wavlm_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, sids, wavlm_padded, wavlm_lengths


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
      # deterministically shuffle based on epoch
      g = torch.Generator()
      g.manual_seed(self.epoch)
  
      indices = []
      if self.shuffle:
          for bucket in self.buckets:
              indices.append(torch.randperm(len(bucket), generator=g).tolist())
      else:
          for bucket in self.buckets:
              indices.append(list(range(len(bucket))))
  
      batches = []
      for i in range(len(self.buckets)):
          bucket = self.buckets[i]
          len_bucket = len(bucket)
          if len_bucket == 0:
              bucket = max(self.buckets, key=len)  # Borrow from a full bucket
              len_bucket = len(bucket)
          ids_bucket = indices[i]
          num_samples_bucket = self.num_samples_per_bucket[i]
  
          # add extra samples to make it evenly divisible
          rem = num_samples_bucket - len_bucket
          ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
  
          # subsample
          ids_bucket = ids_bucket[self.rank::self.num_replicas]
  
          # batching
          for j in range(len(ids_bucket) // self.batch_size):
              batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
              batches.append(batch)
  
      if self.shuffle:
          batch_ids = torch.randperm(len(batches), generator=g).tolist()
          batches = [batches[i] for i in batch_ids]
      self.batches = batches
  
      assert len(self.batches) * self.batch_size == self.num_samples
      return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
      if hi is None:
          hi = len(self.boundaries) - 1
  
      if hi > lo:
          mid = (hi + lo) // 2
          if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
              return mid
          elif x <= self.boundaries[mid]:
              return self._bisect(x, lo, mid)
          else:
              return self._bisect(x, mid + 1, hi)
      else:
          return -1

    def __len__(self):
        return self.num_samples // self.batch_size
