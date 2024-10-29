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

import input_pipeline
from input_pipeline import prepare_batch_data_sqa, apply_mixup_cutmix_batch, pre_process_batch
import models

from utils.display_utils import display_model
from functools import partial

from flax.training.train_state import TrainState as FlaxTrainState
import flax.nnx as nn

import orbax.checkpoint as ocp
from flax.training import checkpoints
import os

NUM_CLASSES = 1000
IMAGE_SIZE = 224

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

def cross_entropy_loss(logits, labels):
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
  return jnp.mean(xentropy)

def compute_metrics(logits, labels):
  # this is the version for both one-hot labels and not one-hot labels, modified by sqa
  # compute per-sample loss
  if labels.shape[-1] != NUM_CLASSES:
    labels = jax.nn.one_hot(labels, NUM_CLASSES)
  loss = optax.softmax_cross_entropy(logits=logits, labels=labels) # (local_batch_size,)
  # print("loss shape: ", loss.shape)  

  accuracy = (jnp.argmax(logits, -1) == jnp.argmax(labels, -1))  # (local_batch_size, )
  metrics = {
    'loss': loss,
    'accuracy': accuracy,
    'labels': labels,
  }
  metrics = lax.all_gather(metrics, axis_name='batch')
  labels = metrics['labels']
  metrics = jax.tree_map(lambda x: x.flatten(), metrics)  # (batch_size,)
  metrics['labels'] = labels
  return metrics


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

def train_step_sqa(state:NNXTrainState, batch, rng_init, learning_rate_fn):
  """Perform a single training step."""
  images, labels = batch['image'], batch['label']

  # ResNet has no dropout; but maintain rng_dropout for future usage
  rng_step = random.fold_in(rng_init, state.step)
  rng_device = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))
  rng_dropout, _ = random.split(rng_device)

  def loss_fn(params):
    """loss function used for training."""
    logits, new_batch_stats, new_rng_params = state.apply_fn(
      state.graphdef, params, state.rng_states, state.batch_stats, True, images) # True: is_training
    loss = cross_entropy_loss(logits, labels)
    return loss, (logits, new_batch_stats, new_rng_params)

  step = state.step
  lr = learning_rate_fn(step)

  grad_fn = nn.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
  grads = lax.pmean(grads, axis_name='batch')

  logits, new_batch_stats, new_rng_params = aux[1]

  metrics = compute_metrics(logits, labels)
  metrics['lr'] = lr

  new_state = state.apply_gradients(
    grads=grads, batch_stats=new_batch_stats, rng_states=new_rng_params
  )

  return new_state, metrics


def eval_step(state:NNXTrainState, batch, rng_init):
  labels = batch['label']
  images = batch['image']

  logits, new_batch_stats, new_rng_params = state.apply_fn(
    state.graphdef, state.params, state.rng_states, state.batch_stats, False, images) # False: is_training
  return compute_metrics(logits, labels)



def restore_checkpoint(state, workdir):
  return checkpoints.restore_checkpoint(workdir, state)


def save_checkpoint(state, workdir):
  state = jax.device_get(jax.tree_util.tree_map(lambda x: x[0], state))
  step = int(state.step)
  logging.info('Saving checkpoint step %d.', step)
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

  if config.optimizer == 'sgd':
    if config.weight_decay != 0.0:
      print("Warning from sqa: weight decay is not supported in SGD")
    if config.grad_norm_clip != "None":
      print("Warning from sqa: grad norm clipping is not supported in SGD")
    tx = optax.sgd(
      learning_rate=learning_rate_fn,
      momentum=config.momentum,
      nesterov=True,
    )
  elif config.optimizer == 'adamw':
    grad_norm_clip = None if config.grad_norm_clip == "None" else config.grad_norm_clip
    no_weight_decay_mask = get_no_weight_decay_dict(params)
    tx = optax.adamw(
      learning_rate=learning_rate_fn,
      b1=0.9,
      b2=0.999,
      eps=1e-8,
      weight_decay=config.weight_decay,
      mask=no_weight_decay_mask,
      # grad_norm_clip=grad_norm_clip, # None if no clipping
    )
  else:
    raise ValueError(f'Unknown optimizer: {config.optimizer}, choose from "sgd" or "adamw"')
  
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
  if rank == 0:
    wandb.init(project='deit_nnx', dir=workdir)
    wandb.config.update(config.to_dict())
  global_seed(config.seed)

  rng = random.key(config.seed)

  image_size = 224

  log_for_0('config.batch_size: {}'.format(config.batch_size))

  ########### Create DataLoaders ###########
  if config.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.batch_size // jax.process_count()
  log_for_0('local_batch_size: {}'.format(local_batch_size))
  log_for_0('jax.local_device_count: {}'.format(jax.local_device_count()))

  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  train_loader, steps_per_epoch = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='train',
  )
  eval_loader, steps_per_eval = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='val',
  )

  assert steps_per_eval > 2


  log_for_0('steps_per_epoch: {}'.format(steps_per_epoch))
  log_for_0('steps_per_eval: {}'.format(steps_per_eval))

  ########### Create Model ###########
  model_cls = getattr(models, config.model)
  rngs = nn.Rngs(config.seed, params=config.seed + 114, dropout=config.seed + 514)
  dtype = get_dtype(config.half_precision)
  model_init_fn = partial(model_cls, num_classes=NUM_CLASSES, dtype=dtype, dropout_rate=config.dropout_rate, stochastic_depth_rate=config.stochastic_depth_rate)
  model = model_init_fn(rngs=rngs)
  display_model(model)

  ########### Create LR FN ###########
  base_learning_rate = config.learning_rate * config.batch_size / 512.0 
  learning_rate_fn = create_learning_rate_fn(config, base_learning_rate, steps_per_epoch)

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
  yierbayiyiliuqi = len(train_loader.dataset) # this equals to 1281167

  # use pmap to parallel training
  p_train_step = jax.pmap(
    functools.partial(train_step_sqa, rng_init=rng, learning_rate_fn=learning_rate_fn),
    axis_name='batch',
  )
  p_eval_step = jax.pmap(
    functools.partial(eval_step, rng_init=rng), axis_name='batch')

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
      batch = pre_process_batch(batch)
      batch = apply_mixup_cutmix_batch(config.dataset, batch)
      step = epoch * steps_per_epoch + (n_batch + 1)
      ep = step * config.batch_size / yierbayiyiliuqi
      # print(batch[0].shape)
      batch = prepare_batch_data_sqa(batch) # shape (num_devices, local_batch_size, 224, 224, 3) 
      assert batch['label'].shape[-1] == NUM_CLASSES
      state, metrics = p_train_step(state, batch) # here is the training step
      if epoch == epoch_offset and n_batch == 0:
        log_for_0(f'Initial compilation takes {timer}. Reset timer.')

      if config.get('log_per_step'):
        if step % config.log_per_step == 0:
          if index == 0:
            tang_reduce(metrics) 
            step_per_sec = config.log_per_step / timer.elapse_with_reset()
            loss_to_display = metrics['loss']
            acc_to_display = metrics['accuracy']
            wandb.log({'train_ep:': ep, 
                        'train_loss': loss_to_display, 'train_accuracy':acc_to_display,
                        'lr': learning_rate_fn(step), 'step': step, 'step_per_sec': step_per_sec})
            log_for_0('epoch: {} step: {} loss: {} accuracy: {}, step_per_sec: {}'.format(ep, step,loss_to_display,acc_to_display,step_per_sec))
      
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
      for n_eval_batch, eval_batch in enumerate(eval_loader):
        if (n_eval_batch + 1) % config.log_per_step == 0:
          if index == 0:
            log_for_0('eval: {}/{}'.format(n_eval_batch + 1, steps_per_eval))
        eval_batch = prepare_batch_data_sqa(eval_batch, local_batch_size)

        metrics = p_eval_step(state, eval_batch)
        tang_reduce(metrics)
        assert metrics['labels'].shape[-1] == NUM_CLASSES
        average_metrics.update(**metrics)

      if index == 0:
        computed_metric = average_metrics.compute()
        use_acc = computed_metric['accuracy']
        use_loss = computed_metric['loss']
        wandb.log({'test_ep:': ep, 
                    'test_loss':use_loss, 'test_accuracy': use_acc,
                      'lr': learning_rate_fn(step), 'step':step})
        log_for_0('epoch: [Eval] {} (step {}) test_loss: {} test_accuracy: {}'.format(ep, step, use_loss, use_acc))
  


  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()
  checkpointer.close() # avoid exiting before checkpt is saved
  if index == 0:
    wandb.finish()

  return state