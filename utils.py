import os
import glob
import sys
import argparse
import datetime
import logging
import json
import random
import re
import socket
import subprocess
import numpy as np
from scipy.io.wavfile import read
import torch

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().setLevel(logging.INFO)
for _noisy_logger in ("matplotlib", "PIL", "numba", "numba.core", "librosa"):
  logging.getLogger(_noisy_logger).setLevel(logging.WARNING)
logger = logging


def _is_valid_tcp_port(port):
  try:
    port_int = int(str(port))
  except (TypeError, ValueError):
    return False
  return 1 <= port_int <= 65535


def _can_bind_tcp_port(host, port):
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      sock.bind((host, int(port)))
    return True
  except OSError:
    return False


def _find_free_tcp_port(host="127.0.0.1", preferred_port=29500):
  if _is_valid_tcp_port(preferred_port) and _can_bind_tcp_port(host, preferred_port):
    return int(preferred_port)

  bind_host = host or "127.0.0.1"
  try:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
      sock.bind((bind_host, 0))
      return int(sock.getsockname()[1])
  except OSError:
    try:
      with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    except OSError:
      if _is_valid_tcp_port(preferred_port):
        return int(preferred_port)
      return 29500


def configure_distributed_environment(default_master_addr="127.0.0.1", preferred_port=29500):
  master_addr = os.environ.get("MASTER_ADDR") or default_master_addr
  os.environ["MASTER_ADDR"] = master_addr

  current_port = os.environ.get("MASTER_PORT")
  if _is_valid_tcp_port(current_port):
    os.environ["MASTER_PORT"] = str(int(current_port))
    return master_addr, os.environ["MASTER_PORT"]

  selected_port = _find_free_tcp_port(master_addr, preferred_port)
  os.environ["MASTER_PORT"] = str(selected_port)
  if current_port:
    logging.getLogger(__name__).warning(
      "Ignoring invalid MASTER_PORT=%r; using free port %s.", current_port, selected_port)
  return master_addr, os.environ["MASTER_PORT"]


def _get_model_state_dict(model):
  if hasattr(model, 'module'):
    return model.module.state_dict()
  return model.state_dict()


def _load_model_state_dict(model, saved_state_dict, strict=False):
  target_model = model.module if hasattr(model, 'module') else model
  if strict:
    target_model.load_state_dict(saved_state_dict)
    return

  current_state_dict = target_model.state_dict()
  new_state_dict = {}
  missing_keys = []
  skipped_keys = []
  for key, value in current_state_dict.items():
    saved_value = saved_state_dict.get(key)
    if saved_value is None:
      missing_keys.append(key)
      new_state_dict[key] = value
      continue
    if hasattr(saved_value, "shape") and hasattr(value, "shape") and saved_value.shape != value.shape:
      skipped_keys.append(key)
      new_state_dict[key] = value
      continue
    new_state_dict[key] = saved_value

  if missing_keys:
    logger.warning("Checkpoint is missing %d model keys; keeping current init for them.", len(missing_keys))
    for key in missing_keys[:20]:
      logger.info("%s is not in the checkpoint", key)
  if skipped_keys:
    logger.warning("Checkpoint has %d shape-mismatched model keys; keeping current init for them.", len(skipped_keys))
    for key in skipped_keys[:20]:
      logger.info("%s has a mismatched checkpoint shape", key)

  target_model.load_state_dict(new_state_dict)


def _atomic_torch_save(payload, checkpoint_path):
  os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)
  tmp_path = checkpoint_path + ".tmp"
  torch.save(payload, tmp_path)
  os.replace(tmp_path, checkpoint_path)


def seed_everything(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def collect_rng_state():
  rng_state = {
    "python": random.getstate(),
    "numpy": np.random.get_state(),
    "torch": torch.get_rng_state()
  }
  if torch.cuda.is_available():
    rng_state["cuda"] = torch.cuda.get_rng_state_all()
  return rng_state


def restore_rng_state(rng_state):
  if not rng_state:
    logger.warning("No RNG state was found in checkpoint; continuing from configured seeds.")
    return

  try:
    if "python" in rng_state:
      random.setstate(rng_state["python"])
    if "numpy" in rng_state:
      np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
      torch.set_rng_state(rng_state["torch"])
    if torch.cuda.is_available() and "cuda" in rng_state:
      torch.cuda.set_rng_state_all(rng_state["cuda"])
  except Exception:
    logger.exception("Failed to restore RNG state from checkpoint.")
    raise


def load_checkpoint(checkpoint_path, model, optimizer=None):
  if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
  iteration = checkpoint_dict.get('iteration', checkpoint_dict.get('epoch', 0))
  learning_rate = checkpoint_dict.get('learning_rate')
  if optimizer is not None and 'optimizer' in checkpoint_dict:
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
  saved_state_dict = checkpoint_dict.get('model')
  if saved_state_dict is None and 'models' in checkpoint_dict:
    saved_state_dict = checkpoint_dict['models'].get('generator')
  if saved_state_dict is None:
    raise KeyError("Checkpoint does not contain a 'model' state dict.")
  _load_model_state_dict(model, saved_state_dict)
  if 'rng_state' in checkpoint_dict:
    restore_rng_state(checkpoint_dict['rng_state'])
  logger.info("Loaded checkpoint '{}' (iteration {})" .format(
    checkpoint_path, iteration))
  return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path,
                    scheduler=None, scaler=None, global_step=None, rng_state=None,
                    extra_state=None):
  logger.info("Saving model and optimizer state at iteration {} to {}".format(
    iteration, checkpoint_path))
  payload = {
    'model': _get_model_state_dict(model),
    'iteration': iteration,
    'learning_rate': learning_rate,
    'rng_state': rng_state if rng_state is not None else collect_rng_state()
  }
  if optimizer is not None:
    payload['optimizer'] = optimizer.state_dict()
  if scheduler is not None:
    payload['scheduler'] = scheduler.state_dict()
  if scaler is not None:
    payload['scaler'] = scaler.state_dict()
  if global_step is not None:
    payload['global_step'] = global_step
  if extra_state is not None:
    payload['extra_state'] = extra_state
  _atomic_torch_save(payload, checkpoint_path)
  
  import os, glob
  dir_name = os.path.dirname(checkpoint_path)
  base_name = os.path.basename(checkpoint_path)
  prefix = base_name.split('_')[0] + '_'
  
  checkpoints = glob.glob(os.path.join(dir_name, prefix + '*.pth'))
  
  valid_checkpoints = []
  for cp in checkpoints:
      suffix = os.path.basename(cp).split('_')[1].split('.')[0]
      if suffix.isdigit():
          valid_checkpoints.append((int(suffix), cp))
          
  valid_checkpoints.sort(key=lambda x: x[0])
  
  if len(valid_checkpoints) > 3:
      for _, cp in valid_checkpoints[:-3]:
          try:
              os.remove(cp)
          except OSError:
              pass


def save_training_state(checkpoint_path, models, optimizers=None, schedulers=None,
                        scaler=None, epoch=0, global_step=0, batch_idx=None,
                        next_epoch=None, next_batch_idx=0, learning_rate=None,
                        metrics=None, extra_state=None):
  logger.info("Saving full training state to %s", checkpoint_path)
  optimizers = optimizers or {}
  schedulers = schedulers or {}
  payload = {
    "format_version": 1,
    "epoch": epoch,
    "iteration": epoch,
    "global_step": global_step,
    "batch_idx": batch_idx,
    "next_epoch": next_epoch if next_epoch is not None else epoch,
    "next_batch_idx": next_batch_idx,
    "learning_rate": learning_rate,
    "models": {name: _get_model_state_dict(model) for name, model in models.items()},
    "optimizers": {name: optimizer.state_dict() for name, optimizer in optimizers.items()},
    "schedulers": {name: scheduler.state_dict() for name, scheduler in schedulers.items()},
    "scaler": scaler.state_dict() if scaler is not None else None,
    "rng_state": collect_rng_state(),
    "metrics": metrics or {},
    "extra_state": extra_state or {}
  }
  _atomic_torch_save(payload, checkpoint_path)


def load_training_state(checkpoint_path, models, optimizers=None, schedulers=None,
                        scaler=None, restore_rng=True, strict=False):
  if not os.path.isfile(checkpoint_path):
    raise FileNotFoundError(checkpoint_path)

  logger.info("Loading full training state from %s", checkpoint_path)
  checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
  optimizers = optimizers or {}
  schedulers = schedulers or {}

  saved_models = checkpoint_dict.get("models", {})
  for name, model in models.items():
    if name not in saved_models:
      raise KeyError("Training checkpoint is missing model '{}'.".format(name))
    _load_model_state_dict(model, saved_models[name], strict=strict)

  saved_optimizers = checkpoint_dict.get("optimizers", {})
  for name, optimizer in optimizers.items():
    if name in saved_optimizers:
      optimizer.load_state_dict(saved_optimizers[name])
    else:
      logger.warning("Training checkpoint is missing optimizer '%s'.", name)

  saved_schedulers = checkpoint_dict.get("schedulers", {})
  for name, scheduler in schedulers.items():
    if name in saved_schedulers:
      scheduler.load_state_dict(saved_schedulers[name])
    else:
      logger.warning("Training checkpoint is missing scheduler '%s'.", name)

  if scaler is not None and checkpoint_dict.get("scaler") is not None:
    scaler.load_state_dict(checkpoint_dict["scaler"])

  if restore_rng:
    restore_rng_state(checkpoint_dict.get("rng_state"))

  logger.info(
    "Loaded training state '%s' (epoch=%s, global_step=%s, next_batch_idx=%s)",
    checkpoint_path,
    checkpoint_dict.get("epoch"),
    checkpoint_dict.get("global_step"),
    checkpoint_dict.get("next_batch_idx")
  )
  return checkpoint_dict


def find_training_checkpoint(model_dir, preferred_name="latest.pt"):
  for name in [preferred_name, "latest.pt", "best.pt", "second_best.pt"]:
    path = os.path.join(model_dir, name)
    if os.path.isfile(path):
      return path
  return None


def hparams_to_dict(hparams):
  if isinstance(hparams, HParams):
    return {key: hparams_to_dict(value) for key, value in hparams.items()}
  if isinstance(hparams, list):
    return [hparams_to_dict(value) for value in hparams]
  return hparams


def _hparam_get(hparams, key, default=None):
  if hparams is None:
    return default
  if isinstance(hparams, dict):
    return hparams.get(key, default)
  return getattr(hparams, key, default)


class WandBRunWrapper:
  def __init__(self, run=None, enabled=False, logger_obj=None):
    self.run = run
    self.enabled = enabled and run is not None
    self.logger = logger_obj or logger

  def log(self, values, step=None):
    if not self.enabled:
      return
    try:
      safe_values = {}
      for key, value in values.items():
        if torch.is_tensor(value):
          value = value.detach().float().mean().cpu().item()
        elif hasattr(value, "item"):
          value = value.item()
        safe_values[key] = value
      self.run.log(safe_values, step=step)
    except Exception:
      self.logger.exception("W&B logging failed; continuing without interrupting training.")
      self.enabled = False

  def watch(self, models, log="gradients", log_freq=1000):
    if not self.enabled:
      return
    try:
      if not isinstance(models, (list, tuple)):
        models = [models]
      for model in models:
        self.run.watch(model, log=log, log_freq=log_freq)
    except Exception:
      self.logger.exception("W&B model watch failed; continuing without watch hooks.")

  def finish(self):
    if not self.enabled:
      return
    try:
      self.run.finish()
    except Exception:
      self.logger.exception("W&B finish failed.")
    finally:
      self.enabled = False


def init_wandb(hps, logger_obj=None):
  logger_obj = logger_obj or logger
  wandb_hps = getattr(hps, "wandb", None)
  env_project = os.environ.get("WANDB_PROJECT")
  enabled = bool(_hparam_get(wandb_hps, "enabled", bool(env_project)))
  if not enabled:
    logger_obj.info("W&B disabled. Set wandb.enabled=true in config or WANDB_PROJECT to enable it.")
    return WandBRunWrapper(logger_obj=logger_obj)

  try:
    import wandb
  except ImportError:
    logger_obj.warning("W&B requested but the wandb package is unavailable; continuing without W&B.")
    return WandBRunWrapper(logger_obj=logger_obj)
  except Exception:
    logger_obj.exception("W&B import failed; continuing without W&B.")
    return WandBRunWrapper(logger_obj=logger_obj)

  try:
    project = _hparam_get(wandb_hps, "project", env_project or "mamba-vits")
    entity = _hparam_get(wandb_hps, "entity", os.environ.get("WANDB_ENTITY"))
    mode = _hparam_get(wandb_hps, "mode", os.environ.get("WANDB_MODE"))
    tags = _hparam_get(wandb_hps, "tags", None)
    notes = _hparam_get(wandb_hps, "notes", None)
    run_name = _hparam_get(wandb_hps, "name", getattr(hps, "run_name", os.path.basename(hps.model_dir)))
    run = wandb.init(
      project=project,
      entity=entity,
      name=run_name,
      dir=hps.model_dir,
      config=hparams_to_dict(hps),
      tags=tags,
      notes=notes,
      mode=mode,
      resume="allow")
    logger_obj.info("W&B initialized: project=%s run=%s", project, run_name)
    return WandBRunWrapper(run=run, enabled=True, logger_obj=logger_obj)
  except Exception:
    logger_obj.exception("W&B initialization failed; continuing without W&B.")
    return WandBRunWrapper(logger_obj=logger_obj)


def summarize(writer, global_step, scalars={}, histograms={}, images={}, audios={}, audio_sampling_rate=22050):
  for k, v in scalars.items():
    writer.add_scalar(k, v, global_step)
  for k, v in histograms.items():
    writer.add_histogram(k, v, global_step)
  for k, v in images.items():
    writer.add_image(k, v, global_step, dataformats='HWC')
  for k, v in audios.items():
    writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
  f_list = glob.glob(os.path.join(dir_path, regex))
  if not f_list:
    raise FileNotFoundError("No checkpoints matched {} in {}".format(regex, dir_path))
  f_list.sort(key=lambda f: (
    int("".join(filter(str.isdigit, os.path.basename(f))) or -1),
    os.path.getmtime(f)
  ))
  x = f_list[-1]
  print(x)
  return x


def _sanitize_run_name(name):
  name = re.sub(r"[^A-Za-z0-9_.-]+", "-", name.strip())
  return name.strip("-") or "model"


def create_experiment_dir(experiment_root, model_name):
  os.makedirs(experiment_root, exist_ok=True)
  timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
  safe_model_name = _sanitize_run_name(model_name)
  base_name = "run_{}_{}".format(timestamp, safe_model_name)
  model_dir = os.path.join(experiment_root, base_name)
  suffix = 1
  while os.path.exists(model_dir):
    suffix += 1
    model_dir = os.path.join(experiment_root, "{}_{:02d}".format(base_name, suffix))
  os.makedirs(model_dir)
  return model_dir


def resolve_experiment_dir(path):
  if os.path.isfile(path):
    return os.path.dirname(path)
  return path


def plot_spectrogram_to_numpy(spectrogram):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np
  
  fig, ax = plt.subplots(figsize=(10,2))
  im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                  interpolation='none')
  plt.colorbar(im, ax=ax)
  plt.xlabel("Frames")
  plt.ylabel("Channels")
  plt.tight_layout()

  fig.canvas.draw()
  fig.canvas.draw()
  data = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def plot_alignment_to_numpy(alignment, info=None):
  global MATPLOTLIB_FLAG
  if not MATPLOTLIB_FLAG:
    import matplotlib
    matplotlib.use("Agg")
    MATPLOTLIB_FLAG = True
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
  import matplotlib.pylab as plt
  import numpy as np

  fig, ax = plt.subplots(figsize=(6, 4))
  im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower',
                  interpolation='none')
  fig.colorbar(im, ax=ax)
  xlabel = 'Decoder timestep'
  if info is not None:
      xlabel += '\n\n' + info
  plt.xlabel(xlabel)
  plt.ylabel('Encoder timestep')
  plt.tight_layout()

  fig.canvas.draw()
  fig.canvas.draw()
  data = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[:, :, :3].copy()
  data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
  plt.close()
  return data


def load_wav_to_torch(full_path):
  sampling_rate, data = read(full_path)
  return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
  with open(filename, encoding='utf-8') as f:
    filepaths_and_text = [line.strip().split(split) for line in f]
  return filepaths_and_text


def get_hparams(init=True):
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--config', type=str, default="./configs/base.json",
                      help='JSON file for configuration')
  parser.add_argument('-m', '--model', type=str, required=True,
                      help='Model name')
  parser.add_argument('--experiment-root', '--experiment_root', dest='experiment_root',
                      type=str, default="./experiments",
                      help='Root directory for timestamped experiment runs')
  parser.add_argument('--resume', type=str, default=None,
                      help='Existing experiment directory or checkpoint file to resume')
  parser.add_argument('--model-dir', '--model_dir', dest='model_dir', type=str, default=None,
                      help='Explicit output directory; use --resume for continuing a prior run')
  
  args = parser.parse_args()
  if args.resume is not None and args.model_dir is not None:
    raise ValueError("Use either --resume or --model-dir, not both.")

  if args.resume is not None:
    model_dir = resolve_experiment_dir(args.resume)
    init = False
  elif args.model_dir is not None:
    model_dir = args.model_dir
  else:
    model_dir = create_experiment_dir(args.experiment_root, args.model)

  if not os.path.exists(model_dir):
    os.makedirs(model_dir)

  config_path = args.config
  config_save_path = os.path.join(model_dir, "config.json")
  if init or not os.path.exists(config_save_path):
    with open(config_path, "r") as f:
      data = f.read()
    with open(config_save_path, "w") as f:
      f.write(data)
  else:
    with open(config_save_path, "r") as f:
      data = f.read()
  config = json.loads(data)
  
  hparams = HParams(**config)
  hparams.model_dir = model_dir
  hparams.experiment_root = args.experiment_root
  hparams.run_name = os.path.basename(model_dir)
  hparams.resume_path = args.resume
  hparams.resume_checkpoint = args.resume if args.resume is not None and os.path.isfile(args.resume) else None
  return hparams


def get_hparams_from_dir(model_dir):
  config_save_path = os.path.join(model_dir, "config.json")
  with open(config_save_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  hparams.model_dir = model_dir
  return hparams


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams


def check_git_hash(model_dir):
  source_dir = os.path.dirname(os.path.realpath(__file__))
  if not os.path.exists(os.path.join(source_dir, ".git")):
    logger.warning("{} is not a git repository, therefore hash value comparison will be ignored.".format(
      source_dir
    ))
    return

  cur_hash = subprocess.getoutput("git rev-parse HEAD")

  path = os.path.join(model_dir, "githash")
  if os.path.exists(path):
    saved_hash = open(path).read()
    if saved_hash != cur_hash:
      logger.warning("git hash values are different. {}(saved) != {}(current)".format(
        saved_hash[:8], cur_hash[:8]))
  else:
    open(path, "w").write(cur_hash)


def get_logger(model_dir, filename="train.log"):
  global logger
  logger_name = os.path.basename(os.path.abspath(model_dir)) or "vits"
  logger = logging.getLogger(logger_name)
  logger.setLevel(logging.DEBUG)
  logger.propagate = False

  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  try:
    if not os.path.exists(model_dir):
      os.makedirs(model_dir)

    log_path = os.path.join(model_dir, filename)
    has_file_handler = any(
      isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == os.path.abspath(log_path)
      for handler in logger.handlers)
    if not has_file_handler:
      file_handler = logging.FileHandler(log_path)
      file_handler.setLevel(logging.DEBUG)
      file_handler.setFormatter(formatter)
      logger.addHandler(file_handler)
  except Exception:
    logging.exception("Could not create file logger at %s; falling back to stdout only.", model_dir)

  has_stream_handler = any(
    isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
    for handler in logger.handlers)
  if not has_stream_handler:
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
  return logger


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()
