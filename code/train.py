# Copied from Kaiming He's resnet_jax repository

import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax

import torch
import numpy as np
import wandb
from utils.logging_util import log_for_0, Timer
from flax import jax_utils as ju
from utils.metric_utils import tang_reduce, MyMetrics, Avger
from torch.utils.data import DataLoader
from utils.utils import train_set, val_set
import ncsnv2

from utils.display_utils import display_model
from functools import partial
from flax.training.train_state import TrainState as FlaxTrainState
import flax.nnx as nn
from kaiming_utils.info_util import print_params
from utils.utils import get_sigmas, save_img, corruption
from langevin import langevin, langevin_masked
from 数据集 import create_split, prepare_batch_data_sqa

import orbax.checkpoint as ocp
from flax.training import checkpoints
import os

class NNXTrainState(FlaxTrainState):
  batch_stats: Any
  rng_states: Any
  graphdef: Any
  # NOTE: is_training can't be a attr, since it can't be replicated

def global_seed(seed):
  torch.manual_seed(seed)
  np.random.seed(seed)
  import random as R
  R.seed(seed)

def get_dtype(half_precision):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_dtype

def create_model(*, model_cls, half_precision, config, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(
    dtype=model_dtype, 
    ngf=config.model.ngf, 
    n_noise_levels=config.sampling.n_noise_levels,
    config=config,
    **kwargs)

def cross_entropy_loss(logits, labels):
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
  return jnp.mean(xentropy)


def create_learning_rate_fn(
  config: ml_collections.ConfigDict,
  base_learning_rate: float,
  steps_per_epoch: int,
):
  """
  Create learning rate schedule.

  first warmup (increase to base_learning_rate) for config.warmup_epochs
  then cosine decay to 0 for the rest of the epochs
  """

  warmup_lr = 1e-6
  warmup_epochs = config.warmup_epochs
  cool_down_lr = 1e-5

  def tang_schedule(step: int) -> float:
    return warmup_lr

  def warmup_schedule(step: int) -> float:
    epoch = step // steps_per_epoch
    return warmup_lr + (base_learning_rate - warmup_lr) * (epoch / warmup_epochs)
    
  def cosine_schedule(step: int) -> float:
    epoch = step // steps_per_epoch
    epoch += warmup_epochs
    current_lr = cool_down_lr + 0.5 * (base_learning_rate - cool_down_lr) * (1 + jnp.cos(jnp.pi * epoch / config.num_epochs))
    return current_lr

  lr_schedule = optax.join_schedules(
    schedules=[tang_schedule, warmup_schedule, cosine_schedule],
    boundaries=[steps_per_epoch, (warmup_epochs+1) * steps_per_epoch]
  )
      
  return lr_schedule

def train_step_sqa(state:NNXTrainState, batch, rng_init, sigmas):
  """Perform a single training step."""

  # ResNet has no dropout; but maintain rng_dropout for future usage
  rng_step = random.fold_in(rng_init, state.step)
  rng_device = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))
  rng, _ = random.split(rng_device)

  images = batch['image']

  sigma_indices = random.randint(rng, (batch['image'].shape[0],), 0, len(sigmas))
  sigma_batch = sigmas[sigma_indices].reshape(-1, 1, 1, 1)

  noise = random.normal(rng, images.shape) * sigma_batch
  images_noise = images + noise
  target = - noise / sigma_batch

  def loss_fn(params):
    """loss function used for training."""
    outputs, new_batch_stats, new_rng_params = state.apply_fn(
      state.graphdef, params, state.rng_states, state.batch_stats, True, images_noise)
    outputs = outputs * sigma_batch
    loss = jnp.mean((outputs - target)**2)
    return loss, (outputs, new_batch_stats, new_rng_params)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = lax.pmean(grads, axis_name='batch')

  outputs, new_batch_stats, new_rng_params = aux[1]

  loss = aux[0]
  loss = lax.pmean(loss, axis_name='batch')
  # TODO: implement ema
  metrics = {"loss": loss}

  new_state = state.apply_gradients(
    grads=grads, batch_stats=new_batch_stats, rng_states=new_rng_params
  )

  return new_state, metrics


def sample_step(state:NNXTrainState, rng_init, sigmas, config, epoch):
  log_for_0(f"start generating samples for epoch {epoch}")
  output = langevin(
    state,
    shape=(64, 1, 28, 28),
    sigmas=sigmas,
    eps=config.eps,
    T=config.T,
    rngs=rng_init,
    whole_process=False,
    clamp=False,
    verbose=True # we will set to False later
  )
  dir=config.save_dir + "generated/"
  save_img(output, dir, im_name=f"{epoch}.png", grid=(8, 8))

  _, all_samples = langevin(
    state,
    shape=(10, 1, 28, 28),
    sigmas=sigmas,
    eps=config.eps,
    T=config.T,
    rngs=rng_init,
    whole_process=True,
    clamp=False,
    verbose=False
  )
  dir=config.save_dir + "sample_process/"
  g = all_samples.shape[0]
  assert g%10 == 0
  save_img(all_samples, dir, im_name=f"{epoch}.png", grid=(g//10, 10)) # this maybe reverse
  log_for_0(f"saved samples for epoch {epoch}")

def denoising_eval_step(state:NNXTrainState, rng_init, sigmas, config, ground_truth, type_, epoch):

  assert type_ in {"even", "lower"}
  ground_truth_0=ground_truth[64:74]
  ground_truth = ground_truth[:64]
  log_for_0(f"evaluating denoising for epoch {epoch}")
  corrupted, mask = corruption(
    ground_truth, 
    type_=type_, 
    rngs=rng_init, 
    noise_scale=1, 
    clamp=False
  )
  # denoising process
  recovered = langevin_masked(
    state,
    x=corrupted,
    sigmas=sigmas,
    eps=config.eps,
    T=config.T,
    mask=mask,
    rngs=rng_init,
    whole_process=False,
    clamp=False,
    verbose=True # we will set to False later
  )
  dir=config.save_dir + f"denoising_{type_}/{epoch}"
  save_img(recovered, dir, im_name=f"recovered.png", grid=(8, 8))
  save_img(ground_truth, dir, im_name=f"groundtruth.png", grid=(8, 8))
  save_img(corrupted, dir, im_name=f"corrupted.png", grid=(8, 8))

  # calculate mse
  mse = jnp.mean((recovered - ground_truth) ** 2)

  corrupted, mask = corruption(
    ground_truth_0, 
    type_=type_, 
    rngs=rng_init, 
    noise_scale=1, 
    clamp=False
  )
  _, all_samples = langevin_masked(
    state,
    x=corrupted,
    sigmas=sigmas,
    eps=config.eps,
    T=config.T,
    mask=mask,
    rngs=rng_init,
    whole_process=True,
    clamp=False,
    verbose=False
  )
  dir=config.save_dir + f"denoising_{type_}_process/{epoch}"
  g = all_samples.shape[0]
  assert g%10 == 0
  save_img(all_samples, dir, im_name=f"recovered.png", grid=(g//10, 10)) # this maybe reverse
  log_for_0(f"saved denoising for epoch {epoch} and type {type_}")

  return mse


def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
  step = int(state.step)
  log_for_0('Saving checkpoint step %d.', step)
  checkpoints.save_checkpoint_multiprocess(workdir, state, step, keep=2)


# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state: NNXTrainState):
  """Sync the batch statistics across replicas. This is called before evaluation."""
  # Each device has its own version of the running average batch statistics and
  if hasattr(state, 'batch_stats'):
    return state
  if len(state.batch_stats) == 0:
      return state
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

def get_no_weight_decay_dict(params):
  def modify_value_based_on_key(obj):
    if not isinstance(obj, dict):
      return obj
    for k,v in obj.items():
      if not isinstance(v,dict):
        if k in {'cls','pos_emb','bias','scale'}:
          obj[k] = False
        else:
          obj[k] = True
    return obj
  def is_leaf(obj):
    if not isinstance(obj, dict):
      return True
    modify_value_based_on_key(obj)
    b = isinstance(obj, dict) and all([not isinstance(v, dict) for v in obj.values()])
    return b
  u = jax.tree_util.tree_map(lambda x:False,params)
  modified_tree = jax.tree_util.tree_map(partial(modify_value_based_on_key), u, is_leaf=is_leaf)
  return modified_tree

def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
  """
  Create initial training state, including the model and optimizer.
  """
  # print("here we are in the function 'create_train_state' in train.py; ready to define optimizer")
  graphdef, params, batch_stats, rng_states = nn.split(model, nn.Param, nn.BatchStat, nn.RngState)

  def apply_fn(graphdef2, params2, rng_states2, batch_stats2, is_training, x):
    merged_model = nn.merge(graphdef2, params2, rng_states2, batch_stats2)
    if is_training:
      merged_model.train()
    else:
      merged_model.eval()
    del params2, rng_states2, batch_stats2
    out = merged_model(x)
    new_batch_stats, new_rng_states, _ = nn.state(merged_model, nn.BatchStat, nn.RngState, ...)
    return out, new_batch_stats, new_rng_states

  # here is the optimizer

  # if config.optimizer == 'sgd':
  #   if config.weight_decay != 0.0:
  #     print("Warning from sqa: weight decay is not supported in SGD")
  #   if config.grad_norm_clip != "None":
  #     print("Warning from sqa: grad norm clipping is not supported in SGD")
  #   tx = optax.sgd(
  #     learning_rate=learning_rate_fn,
  #     momentum=config.momentum,
  #     nesterov=True,
  #   )
  # elif config.optimizer == 'adamw':
  #   grad_norm_clip = None if config.grad_norm_clip == "None" else config.grad_norm_clip
  #   tx = optax.adamw(
  #     learning_rate=learning_rate_fn,
  #     b1=0.9,
  #     b2=0.999,
  #     eps=1e-8,
  #     weight_decay=config.weight_decay,
  #     # grad_norm_clip=grad_norm_clip, # None if no clipping
  #   )
  # else:
  #   raise ValueError(f'Unknown optimizer: {config.optimizer}, choose from "sgd" or "adamw"')
  # use adamw optimizer
  tx = optax.adamw(
    learning_rate=learning_rate_fn,
    b1=0.9,
    b2=0.999,
    eps=1e-8,
    weight_decay=config.weight_decay,
  )
  state = NNXTrainState.create(
    graphdef=graphdef,
    apply_fn=apply_fn,
    params=params,
    tx=tx,
    batch_stats=batch_stats,
    rng_states=rng_states,
  )
  return state

def _update_model_avg(model_avg, state_params, ema_decay):
  return jax.tree_util.tree_map(lambda x, y: ema_decay * x + (1 - ema_decay) * y, model_avg, state_params)
  # return model_avg

def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> NNXTrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  ########### Initialize ###########
  rank = index = jax.process_index()
  print("rank: ", rank) # we hope this is 0-31
  if rank == 0:
    wandb.init(project='deit_nnx', dir=workdir)
    wandb.config.update(config.to_dict())
  global_seed(config.seed)

  rng = random.key(config.seed)

  image_size = config.dataset.image_size
  assert image_size == 28

  log_for_0('config.batch_size: {}'.format(config.batch_size))

  ########### Create DataLoaders ###########
  if config.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.batch_size // jax.process_count()
  log_for_0('local_batch_size: {}'.format(local_batch_size))
  log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))

  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  train_set = train_set(root=config.dataset.root)
  val_set = val_set(root=config.dataset.root)

  train_loader, steps_per_epoch = create_split(
    train_set, local_batch_size, 'train', config
  )

  # eval_loader, steps_per_eval = create_split(
  #   val_set, local_batch_size, 'val', config
  # )

  eval_loader = DataLoader(val_set, batch_size=config.eval_batch_size, shuffle=True, drop_last=False, pin_memory=True)
  # steps_per_eval = len(eval_loader)
  steps_per_eval = 4

  log_for_0('steps_per_epoch: {}'.format(steps_per_epoch))
  log_for_0('steps_per_eval: {}'.format(steps_per_eval))

  if config.steps_per_eval != -1:
    steps_per_eval = config.steps_per_eval

  ########### Create Model ###########
  model_cls = getattr(ncsnv2, config.model)
  rngs = nn.Rngs(config.seed, params=config.seed + 114, dropout=config.seed + 514, evaluation=config.seed + 1919)
  dtype = get_dtype(config.half_precision)
  # model = create_model(
  #   model_cls=model_cls, half_precision=config.half_precision,
  #   config=config
  # )
  model_init_fn = partial(
    model_cls, 
    dtype=dtype, 
    ngf=config.ngf, 
    n_noise_levels=config.n_noise_levels, 
    config=config
  )
  model = model_init_fn(rngs=rngs)
  display_model(model)

  ########### Create LR FN ###########
  # learning_rate_fn = create_learning_rate_fn(config, base_learning_rate, steps_per_epoch)
  learning_rate_fn = config.learning_rate

  ########### Create Train State ###########
  state = create_train_state(rng, config, model, image_size, learning_rate_fn)
  # restore checkpoint
  if config.load_from is not None:
    if not os.path.isabs(config.load_from):
      raise ValueError('Checkpoint path must be absolute')
    if not os.path.exists(config.load_from):
      raise ValueError('Checkpoint path {} does not exist'.format(config.load_from))
    state = restore_checkpoint(model_init_fn ,state, config.load_from)
    # sanity check, as in Kaiming's code
    assert state.step > 0 and state.step % steps_per_epoch == 0, ValueError('Got an invalid checkpoint with step {}'.format(state.step))
  epoch_offset = state.step // steps_per_epoch  # sanity check for resuming

  state = ju.replicate(state) # NOTE: this doesn't split the RNGs automatically, but it is an intended behavior
  model_avg = state.params
  yierbayiyiliuqi = len(train_loader.dataset) # this equals to ??
  print("yierbayiyiliuqi: ", yierbayiyiliuqi)

  # use pmap to parallel training
  sigmas = get_sigmas(config)
  p_train_step = jax.pmap(
    functools.partial(train_step_sqa, rng_init=rng, sigmas=sigmas),
    axis_name='batch',
  )

  ########### Checkpointer ###########
  checkpointer = ocp.StandardCheckpointer()
  def _restore(ckpt_path, item, **restore_kwargs):
      return ocp.StandardCheckpointer.restore(checkpointer, ckpt_path, target=item)
  setattr(checkpointer, 'restore', _restore)
  def save_checkpoint(state:NNXTrainState, workdir):
      # TODO: this function currently emits lots of "background messages". Try to suppress them
      state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
      step = int(state.step)
      log_for_0('Saving checkpoint to {}, with step {}'.format(workdir, step))
      merged_params: nn.State = state.params
      # 不能把rng merge进去！
      # if len(state.rng_states) > 0:
      #     merged_params = nn.State.merge(merged_params, state.rng_states)
      if len(state.batch_stats) > 0:
          merged_params = nn.State.merge(merged_params, state.batch_stats)
      checkpoints.save_checkpoint_multiprocess(workdir, {
          'mo_xing': merged_params,
          'you_hua_qi': state.opt_state,
          'step': step
      }, step, keep=2, orbax_checkpointer=checkpointer)
      # TODO: FATAL: this "keep" param seems not being used. This must be fixed ASAP!
  def restore_checkpoint(model_init_fn, state, workdir):
      abstract_model = nn.eval_shape(lambda: model_init_fn(rngs=nn.Rngs(0)))
      rng_states = state.rng_states
      abs_state = nn.state(abstract_model)
      _, useful_abs_state = abs_state.split(nn.RngState, ...)
      fake_state = {
          'mo_xing': useful_abs_state,
          'you_hua_qi': state.opt_state,
          'step': 0
      }
      loaded_state = checkpoints.restore_checkpoint(workdir, target=fake_state,orbax_checkpointer=checkpointer)
      merged_params = loaded_state['mo_xing']
      opt_state = loaded_state['you_hua_qi']
      step = loaded_state['step']
      params, batch_stats = merged_params.split(nn.Param, nn.BatchStat)
      return state.replace(
          params=params,
          rng_states=rng_states,
          batch_stats=batch_stats,
          opt_state=opt_state,
          step=step
      )

  ########### Training Loop ###########
  log_for_0('Initial compilation, this might take some minutes...')

  last_model = None
  if config.get('ema_decay'):
    assert config.ema_decay > 0.0 and config.ema_decay < 1.0, 'ema_decay should be in (0, 1)'
    log_for_0('Using EMA with decay {}'.format(config.ema_decay))
    p_update_model_avg = jax.pmap(partial(_update_model_avg, ema_decay=config.ema_decay), axis_name='batch')

  for epoch in range(epoch_offset, config.num_epochs):
    ########### Train ###########
    timer = Timer()
    if jax.process_count() > 1:
      train_loader.sampler.set_epoch(epoch)
    log_for_0('epoch {}...'.format(epoch))
    timer.reset()
    for n_batch, batch in enumerate(train_loader):

      images = batch[0].reshape(-1, config.dataset.channels, config.dataset.image_size, config.dataset.image_size)
      step = epoch * steps_per_epoch + n_batch
      ep = step * config.batch_size / yierbayiyiliuqi
      print("images.shape: ", images.shape)
      images = prepare_batch_data_sqa(images)

      # print("batch['image'].shape:", batch['image'].shape)
      # assert False

      # # here is code for us to visualize the images
      # import matplotlib.pyplot as plt
      # import numpy as np
      # import os
      # print(batch["image"].shape)

      # # save batch["image"] to ./images/{epoch}/i.png
      # rank = jax.process_index()

      # if os.path.exists(f"/kmh-nfs-us-mount/staging/sqa/images/{n_batch}/{rank}") == False:
      #   os.makedirs(f"/kmh-nfs-us-mount/staging/sqa/images/{n_batch}/{rank}")
      # for i in range(batch["image"][0].shape[0]):
      #   # print the max and min of the image
      #   # print(f"max: {np.max(batch['image'][0][i])}, min: {np.min(batch['image'][0][i])}")
      #   # use the max and min to normalize the image to [0, 1]
      #   img = batch["image"][0][i]
      #   img = (img - np.min(img)) / (np.max(img) - np.min(img))
      #   plt.imsave(f"/kmh-nfs-us-mount/staging/sqa/images/{n_batch}/{rank}/{i}.png", img)
      #   # if i>6: break

      # print(f"saving images for n_batch {n_batch}, done.")
      # if n_batch > 0:
      #   exit(114514)
      # continue


      # print(batch["image"].shape)

      state, metrics = p_train_step(state, batch) # here is the training step
      
      if epoch == epoch_offset and n_batch == 0:
        log_for_0('Initial compilation completed. Reset timer.')

      if config.get('log_per_step'):
        if (step + 1) % config.log_per_step == 0:
          if index == 0:
            tang_reduce(metrics) 
            step_per_sec = config.log_per_step / timer.elapse_with_reset()
            loss_to_display = metrics['loss']
            wandb.log({'train_ep:': ep, 
                        'train_loss': loss_to_display, 
                        # 'lr': learning_rate_fn(step), 
                        'step': step, 
                        'step_per_sec': step_per_sec})
            log_for_0('epoch: {} step: {} loss: {}, step_per_sec: {}'.format(ep, step, loss_to_display, step_per_sec))
      if config.get('ema_decay'):
        # EMA
        model_avg = p_update_model_avg(model_avg, state.params)

    ########### Save Checkpt ###########
    # we first save checkpoint, then do eval. Reasons: 1. if eval emits an error, then we still have our model; 2. avoid the program exits before the checkpointer finishes its job.
    # NOTE: when saving checkpoint, should sync batch stats first.
    state = sync_batch_stats(state)
    if (epoch + 1) % config.checkpoint_per_epoch == 0:
        # if index == 0:
        save_checkpoint(state, workdir)
    if epoch == config.num_epochs - 1:
      state = state.replace(params=model_avg)

    ########### Eval ###########
    if (epoch + 1) % config.eval_per_epoch == 0:
      log_for_0('Eval epoch {}...'.format(epoch))
      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      average_metrics = MyMetrics(reduction=Avger)
      sample_step(state, rng, sigmas, config, epoch)
      for n_eval_batch, eval_batch in enumerate(eval_loader):
        images = eval_batch[0].reshape(-1, config.dataset.channels, config.dataset.image_size, config.dataset.image_size)
        ground_truth = prepare_batch_data_sqa(images)
        if n_eval_batch == 0:
          mse_lower = denoising_eval_step(state, rng, sigmas, config, ground_truth, "lower", epoch)
        if n_eval_batch == 1:
          mse_even = denoising_eval_step(state, rng, sigmas, config, ground_truth, "even", epoch)
          break
        # if (n_eval_batch + 1) % config.log_per_step == 0:
        #   if index == 0:
        #     log_for_0('eval: {}/{}'.format(n_eval_batch + 1, steps_per_eval))
        # eval_image = prepare_batch_data_sqa(eval_batch[0], local_batch_size)

        # metrics = p_eval_step(state, eval_batch) # here is the eval step
        # # print("metrics' labels shape:", metrics['labels'].shape)
        # assert metrics['labels'].shape[-1] == NUM_CLASSES
        # eval_metrics.append(metrics)
      print("mse_lower: ", mse_lower)
      mse_lower = jnp.mean(mse_lower)
      mse_even = jnp.mean(mse_even)

      if index == 0:
        wandb.log({'mse_lower': mse_lower, 'mse_even': mse_even, 'epoch': epoch})
        log_for_0('epoch: {}; mse_lower: {}, mse_even: {}'.format(epoch, mse_lower, mse_even))


  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  checkpointer.close() # avoid exiting before checkpt is saved
  if index == 0:
    wandb.finish()

  return state