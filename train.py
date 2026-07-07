import os
import json
import argparse
import itertools
import math
import shutil  
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
try:
  from tqdm import tqdm
except Exception:
  tqdm = None

import commons
import utils
from data_utils import (
  TextAudioLoader,
  TextAudioCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from text.symbols import symbols


torch.backends.cudnn.benchmark = True
global_step = 0
best_loss = float('inf')          
second_best_loss = float('inf')   


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."

  n_gpus = torch.cuda.device_count()
  utils.configure_distributed_environment()

  hps = utils.get_hparams()
  mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
  global global_step, best_loss, second_best_loss
  wandb_run = None
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    logger.info(
      "Distributed init: MASTER_ADDR=%s MASTER_PORT=%s WORLD_SIZE=%s",
      os.environ.get("MASTER_ADDR"),
      os.environ.get("MASTER_PORT"),
      n_gpus)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))
    wandb_run = utils.init_wandb(hps, logger)

  dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  utils.seed_everything(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioLoader(hps.data.training_files, hps.data)
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioCollate()
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioLoader(hps.data.validation_files, hps.data)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False,
        batch_size=hps.train.batch_size, pin_memory=True,
        drop_last=False, collate_fn=collate_fn)

  net_g = SynthesizerTrn(
      len(symbols),
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
  optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
  net_g = DDP(net_g, device_ids=[rank])
  net_d = DDP(net_d, device_ids=[rank])
  if rank == 0 and wandb_run is not None:
    wandb_run.watch([net_g.module, net_d.module], log_freq=getattr(hps.train, "log_interval", 1000))

  epoch_str = 1
  resume_batch_idx = 0
  if "resume_checkpoint" in hps and hps.resume_checkpoint:
    training_checkpoint_path = hps.resume_checkpoint
  else:
    training_checkpoint_path = utils.find_training_checkpoint(hps.model_dir, "latest.pt")

  scaler = GradScaler(enabled=hps.train.fp16_run)
  is_full_training_checkpoint = (
    training_checkpoint_path is not None and
    os.path.basename(training_checkpoint_path) in ["latest.pt", "best.pt", "second_best.pt"]
  )
  if is_full_training_checkpoint:
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay)
    try:
      training_state = utils.load_training_state(
        training_checkpoint_path,
        models={"generator": net_g, "discriminator": net_d},
        optimizers={"generator": optim_g, "discriminator": optim_d},
        schedulers={"generator": scheduler_g, "discriminator": scheduler_d},
        scaler=scaler)
      epoch_str = int(training_state.get("next_epoch", training_state.get("epoch", 1)))
      resume_batch_idx = int(training_state.get("next_batch_idx", 0) or 0)
      global_step = int(training_state.get("global_step", 0) or 0)
      metrics = training_state.get("metrics", {})
      best_loss = float(metrics.get("best_loss", best_loss))
      second_best_loss = float(metrics.get("second_best_loss", second_best_loss))
      if rank == 0:
        logger.info("Resuming from %s at epoch=%s batch=%s global_step=%s",
                    training_checkpoint_path, epoch_str, resume_batch_idx, global_step)
    except Exception:
      if rank == 0:
        logger.exception("Failed to restore full training checkpoint: %s", training_checkpoint_path)
      raise
  else:
    try:
      if os.path.exists(os.path.join(hps.model_dir, "G_last.pth")):
        _, _, _, epoch_str = utils.load_checkpoint(os.path.join(hps.model_dir, "G_last.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(os.path.join(hps.model_dir, "D_last.pth"), net_d, optim_d)
      else:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
      global_step = (epoch_str - 1) * len(train_loader)
      if rank == 0:
        logger.info("Resumed from legacy G/D checkpoints at epoch=%s global_step=%s", epoch_str, global_step)
    except Exception as exc:
      epoch_str = 1
      global_step = 0
      if rank == 0:
        logger.info("No checkpoint restored; starting a fresh run. Reason: %s", exc)
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    skip_batches = resume_batch_idx if epoch == epoch_str else 0
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval], skip_batches, wandb_run)
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None, skip_batches, None)
    resume_batch_idx = 0
    scheduler_g.step()
    scheduler_d.step()

  if rank == 0:
    writer.close()
    writer_eval.close()
    if wandb_run is not None:
      wandb_run.finish()

def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers, skip_batches=0, wandb_run=None):
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step, best_loss, second_best_loss

  net_g.train()
  net_d.train()
  if skip_batches > 0 and rank == 0:
    logger.info("Skipping %s already-processed batches in epoch %s after checkpoint resume.", skip_batches, epoch)

  train_iter = enumerate(train_loader)
  progress_bar = None
  if tqdm is not None:
    progress_bar = tqdm(
      train_iter,
      total=len(train_loader),
      desc="Epoch {}".format(epoch),
      disable=(rank != 0),
      dynamic_ncols=True)
    train_iter = progress_bar
  elif rank == 0:
    logger.warning("tqdm is not installed; progress bars are disabled.")

  for batch_idx, batch in train_iter:
    if batch_idx < skip_batches:
      continue

    x, x_lengths, spec, spec_lengths, y, y_lengths = batch
    x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
    spec, spec_lengths = spec.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
    y, y_lengths = y.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)

    with autocast(enabled=hps.train.fp16_run):
      y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
      (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths)

      mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax)
      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )

      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) 

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      with autocast(enabled=False):
        loss_dur = torch.sum(l_length.float())
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

        loss_fm = feature_loss(fmap_r, fmap_g)
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_dur, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        if progress_bar is not None:
          progress_bar.set_postfix({
            "step": global_step,
            "g": "{:.3f}".format(loss_gen_all.item()),
            "d": "{:.3f}".format(loss_disc_all.item()),
            "mel": "{:.3f}".format(loss_mel.item()),
            "lr": "{:.2e}".format(lr)
          })
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
            "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)
        if wandb_run is not None:
          wandb_run.log(scalar_dict, step=global_step)

      if global_step % hps.train.eval_interval == 0:
        val_loss = evaluate(hps, net_g, eval_loader, writer_eval)
        if wandb_run is not None:
          wandb_run.log({"val/mel_loss": val_loss, "epoch": epoch}, step=global_step)

        next_batch_idx = batch_idx + 1
        next_epoch = epoch
        if next_batch_idx >= len(train_loader):
          # Resume through an empty tail of this epoch so scheduler.step() stays in order.
          next_batch_idx = len(train_loader)
        next_global_step = global_step + 1

        def save_full_state(filename):
          metrics = {
            "best_loss": best_loss,
            "second_best_loss": second_best_loss,
            "last_val_loss": val_loss
          }
          utils.save_training_state(
            os.path.join(hps.model_dir, filename),
            models={"generator": net_g, "discriminator": net_d},
            optimizers={"generator": optim_g, "discriminator": optim_d},
            schedulers={"generator": scheduler_g, "discriminator": scheduler_d},
            scaler=scaler,
            epoch=epoch,
            global_step=next_global_step,
            batch_idx=batch_idx,
            next_epoch=next_epoch,
            next_batch_idx=next_batch_idx,
            learning_rate=optim_g.param_groups[0]['lr'],
            metrics=metrics)

        try:
          # Full bundles are the authoritative training-resume checkpoints.
          if val_loss < best_loss:
            previous_best_loss = best_loss
            if os.path.exists(os.path.join(hps.model_dir, "best.pt")):
              shutil.copy2(os.path.join(hps.model_dir, "best.pt"), os.path.join(hps.model_dir, "second_best.pt"))
            if os.path.exists(os.path.join(hps.model_dir, "G_best.pth")):
              shutil.copy2(os.path.join(hps.model_dir, "G_best.pth"), os.path.join(hps.model_dir, "G_second_best.pth"))
            if os.path.exists(os.path.join(hps.model_dir, "D_best.pth")):
              shutil.copy2(os.path.join(hps.model_dir, "D_best.pth"), os.path.join(hps.model_dir, "D_second_best.pth"))

            best_loss = val_loss
            second_best_loss = previous_best_loss
            save_full_state("best.pt")
            utils.save_checkpoint(net_g, optim_g, optim_g.param_groups[0]['lr'], epoch, os.path.join(hps.model_dir, "G_best.pth"),
                                  scheduler=scheduler_g, scaler=scaler, global_step=next_global_step)
            utils.save_checkpoint(net_d, optim_d, optim_d.param_groups[0]['lr'], epoch, os.path.join(hps.model_dir, "D_best.pth"),
                                  scheduler=scheduler_d, scaler=scaler, global_step=next_global_step)
            logger.info(f"--> [NEW CHAMPION] Step {global_step}: best.pt updated (Val Mel Loss: {best_loss:.4f})")

          elif val_loss < second_best_loss:
            second_best_loss = val_loss
            save_full_state("second_best.pt")
            utils.save_checkpoint(net_g, optim_g, optim_g.param_groups[0]['lr'], epoch, os.path.join(hps.model_dir, "G_second_best.pth"),
                                  scheduler=scheduler_g, scaler=scaler, global_step=next_global_step)
            utils.save_checkpoint(net_d, optim_d, optim_d.param_groups[0]['lr'], epoch, os.path.join(hps.model_dir, "D_second_best.pth"),
                                  scheduler=scheduler_d, scaler=scaler, global_step=next_global_step)
            logger.info(f"--> [RUNNER-UP CHOSEN] Step {global_step}: second_best.pt updated (Val Mel Loss: {second_best_loss:.4f})")

          save_full_state("latest.pt")
          utils.save_checkpoint(net_g, optim_g, optim_g.param_groups[0]['lr'], epoch, os.path.join(hps.model_dir, "G_last.pth"),
                                scheduler=scheduler_g, scaler=scaler, global_step=next_global_step)
          utils.save_checkpoint(net_d, optim_d, optim_d.param_groups[0]['lr'], epoch, os.path.join(hps.model_dir, "D_last.pth"),
                                scheduler=scheduler_d, scaler=scaler, global_step=next_global_step)
        except Exception:
          logger.exception("Checkpoint save failed at epoch=%s batch=%s global_step=%s", epoch, batch_idx, global_step)
          raise
          
        # 3. Disk Space Cleaner: Identify and purge any lingering step-numbered models
        for filename in os.listdir(hps.model_dir):
          if (filename.startswith("G_") or filename.startswith("D_")) and filename.endswith(".pth"):
            if filename not in ["G_last.pth", "D_last.pth", "G_best.pth", "D_best.pth", "G_second_best.pth", "D_second_best.pth"]:
              try:
                os.remove(os.path.join(hps.model_dir, filename))
              except Exception:
                pass
                
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    val_mel_loss = 0.0
    val_count = 0
    
    with torch.no_grad():
      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        
        y_hat, l_length, attn, ids_slice, x_mask, z_mask, _ = generator(x, x_lengths, spec, spec_lengths)
        
        mel = spec_to_mel_torch(
            spec, 
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate,
            hps.data.mel_fmin, 
            hps.data.mel_fmax)
        y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        y_hat_mel = mel_spectrogram_torch(
            y_hat.squeeze(1), 
            hps.data.filter_length, 
            hps.data.n_mel_channels, 
            hps.data.sampling_rate, 
            hps.data.hop_length, 
            hps.data.win_length, 
            hps.data.mel_fmin, 
            hps.data.mel_fmax
        )
        
        loss_mel = F.l1_loss(y_mel, y_hat_mel)
        val_mel_loss += loss_mel.item()
        val_count += 1

      avg_val_loss = val_mel_loss / val_count if val_count > 0 else float('inf')

      for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(eval_loader):
        x, x_lengths = x.cuda(0), x_lengths.cuda(0)
        spec, spec_lengths = spec.cuda(0), spec_lengths.cuda(0)
        y, y_lengths = y.cuda(0), y_lengths.cuda(0)

        x = x[:1]
        x_lengths = x_lengths[:1]
        spec = spec[:1]
        spec_lengths = spec_lengths[:1]
        y = y[:1]
        y_lengths = y_lengths[:1]
        break
        
      y_hat, attn, mask, *_ = generator.module.infer(x, x_lengths, max_len=1000)
      y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
      
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0,:,:y_hat_lengths[0]]
    }
    if global_step == 0:
      image_dict.update({"gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())})
      audio_dict.update({"gt/audio": y[0,:,:y_lengths[0]]})

    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()
    
    return avg_val_loss

                           
if __name__ == "__main__":
  main()
