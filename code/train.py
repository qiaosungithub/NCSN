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
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
from jax import random
import ml_collections
import optax

import input_pipeline
from input_pipeline import prepare_batch_data, prepare_batch_data_sqa, apply_mixup_cutmix_batch, pre_process_batch
import models

import utils.writer_util as writer_util  # must be after 'from clu import metric_writers'
from utils.info_util import print_params


NUM_CLASSES = 1000


def create_model(*, model_cls, half_precision, **kwargs):
  platform = jax.local_devices()[0].platform
  if half_precision:
    if platform == 'tpu':
      model_dtype = jnp.bfloat16
    else:
      model_dtype = jnp.float16
  else:
    model_dtype = jnp.float32
  return model_cls(num_classes=NUM_CLASSES, dtype=model_dtype, **kwargs)


def initialized(key, image_size, model):
  input_shape = (1, image_size, image_size, 3)

  @jax.jit
  def init(*args):
    return model.init(*args, rng=key)

  logging.info('Initializing params...')
  variables = init({'params': key}, jnp.ones(input_shape, model.dtype))
  if 'batch_stats' not in variables:
    variables['batch_stats'] = {}
  logging.info('Initializing params done.')
  return variables['params'], variables['batch_stats']


def cross_entropy_loss(logits, labels, label_smoothing=0.1):
  # one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
  # apply label smoothing
  smooth_labels = optax.smooth_labels(labels, alpha=label_smoothing)
  xentropy = optax.softmax_cross_entropy(logits=logits, labels=smooth_labels)
  return jnp.mean(xentropy)


# def compute_metrics(logits, labels):
#   # compute per-sample loss
#   one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
#   xentropy = optax.softmax_cross_entropy(logits=logits, labels=one_hot_labels)
#   loss = xentropy  # (local_batch_size,)

#   accuracy = (jnp.argmax(logits, -1) == labels)  # (local_batch_size,)
#   metrics = {
#       'loss': loss,
#       'accuracy': accuracy,
#       'labels': labels,
#   }
#   metrics = lax.all_gather(metrics, axis_name='batch')
#   metrics = jax.tree_map(lambda x: x.flatten(), metrics)  # (batch_size,)
#   return metrics

def compute_metrics(logits, labels):
  # this is the version for both one-hot labels and not one-hot labels
  # compute per-sample loss
  # one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
  # print("labels.shape:", labels.shape)
  if labels.shape[-1] != NUM_CLASSES:
    labels = jax.nn.one_hot(labels, NUM_CLASSES)

  xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
  loss = xentropy  # (local_batch_size,)

  accuracy = (jnp.argmax(logits, -1) == jnp.argmax(labels, -1))  # (local_batch_size, )
  # here we modify, but not very well defined
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
      'labels': labels,
  }
  # print("1 metrics' labels shape:", metrics['labels'].shape)
  metrics = lax.all_gather(metrics, axis_name='batch')
  labels = metrics['labels']
  metrics = jax.tree_map(lambda x: x.flatten(), metrics)  # (batch_size,)
  metrics['labels'] = labels
  # print("2 metrics' labels shape:", metrics['labels'].shape)
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
  # warmup_fn = optax.linear_schedule(
  #     init_value=0.0,
  #     end_value=base_learning_rate,
  #     transition_steps=config.warmup_epochs * steps_per_epoch,
  # )
  cosine_epochs = max((config.num_epochs-10) - config.warmup_epochs, 1)
  # cosine_fn = optax.cosine_decay_schedule(
  #     init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
  # )
  # schedule_fn = optax.join_schedules(
  #     schedules=[warmup_fn, cosine_fn],
  #     boundaries=[config.warmup_epochs * steps_per_epoch],
  # )
  # print('warmup_epochs:', config.warmup_epochs)
  # print('cosine_epochs:', cosine_epochs)
  # print('steps_per_epoch:', steps_per_epoch)
  first_schedule = optax.schedules.warmup_cosine_decay_schedule(init_value=0.0, peak_value=base_learning_rate, warmup_steps=config.warmup_epochs*steps_per_epoch, decay_steps=(config.warmup_epochs + cosine_epochs)*steps_per_epoch, end_value=1e-5) 
  second_schedule = optax.schedules.constant_schedule(value=1e-5)
  return optax.join_schedules(schedules=[first_schedule, second_schedule], boundaries=[(config.warmup_epochs + cosine_epochs)*steps_per_epoch])


# def train_step(state, batch, rng_init, learning_rate_fn,label_smoothing=0.1):
#   """Perform a single training step."""

#   # ResNet has no dropout; but maintain rng_dropout for future usage
#   rng_step = random.fold_in(rng_init, state.step)
#   rng_device = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))
#   rng_dropout, _ = random.split(rng_device)

#   def loss_fn(params):
#     """loss function used for training."""
#     logits, new_model_state = state.apply_fn(
#         {'params': params, 'batch_stats': state.batch_stats},
#         batch['image'],
#         mutable=['batch_stats'],
#         rngs=dict(dropout=rng_dropout),
#     )
#     loss = cross_entropy_loss(logits, batch['label'], label_smoothing=label_smoothing)
#     # weight_penalty_params = jax.tree_util.tree_leaves(params)
#     # weight_decay = 0.0001
#     # weight_l2 = sum(
#     #     jnp.sum(x**2) for x in weight_penalty_params if x.ndim > 1
#     # )
#     # weight_penalty = weight_decay * 0.5 * weight_l2
#     # loss = loss + weight_penalty
#     return loss, (new_model_state, logits)

#   step = state.step
#   dynamic_scale = state.dynamic_scale
#   lr = learning_rate_fn(step)

#   if dynamic_scale:
#     grad_fn = dynamic_scale.value_and_grad(
#         loss_fn, has_aux=True, axis_name='batch'
#     )
#     dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
#     # dynamic loss takes care of averaging gradients across replicas
#   else:
#     grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
#     aux, grads = grad_fn(state.params)
#     # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
#     grads = lax.pmean(grads, axis_name='batch')
#   new_model_state, logits = aux[1]
#   metrics = compute_metrics(logits, batch['label'])
#   metrics['lr'] = lr

#   new_state = state.apply_gradients(
#       grads=grads, batch_stats=new_model_state['batch_stats']
#   )
#   if dynamic_scale:
#     # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
#     # params should be restored (= skip this step).
#     new_state = new_state.replace(
#         opt_state=jax.tree_util.tree_map(
#             functools.partial(jnp.where, is_fin),
#             new_state.opt_state,
#             state.opt_state,
#         ),
#         params=jax.tree_util.tree_map(
#             functools.partial(jnp.where, is_fin), new_state.params, state.params
#         ),
#         dynamic_scale=dynamic_scale,
#     )
#     metrics['scale'] = dynamic_scale.scale

#   return new_state, metrics

def train_step_sqa(state, batch, rng_init, learning_rate_fn,label_smoothing=0.1):
  """Perform a single training step."""

  # ResNet has no dropout; but maintain rng_dropout for future usage
  rng_step = random.fold_in(rng_init, state.step)
  rng_device = random.fold_in(rng_step, lax.axis_index(axis_name='batch'))
  rng_dropout, _ = random.split(rng_device)

  def categorical_cross_entropy_loss(logits, labels,label_smoothing=label_smoothing):
    """计算分类交叉熵损失"""
    # one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
    # xentropy = optax.softmax_cross_entropy(logits=logits, labels=labels)
    # return jnp.mean(xentropy)
    return cross_entropy_loss(logits, labels,label_smoothing=label_smoothing)

  def loss_fn(params):
    """loss function used for training."""
    logits, new_model_state = state.apply_fn(
      {'params': params, 'batch_stats': state.batch_stats},
      batch['image'],
      mutable=['batch_stats'],
      # rngs=dict(dropout=rng_dropout),
      rng=rng_dropout,
    )
    loss = categorical_cross_entropy_loss(logits, batch['label'])
    return loss, (new_model_state, logits)

  step = state.step
  dynamic_scale = state.dynamic_scale
  lr = learning_rate_fn(step)

  if dynamic_scale:
    grad_fn = dynamic_scale.value_and_grad(
      loss_fn, has_aux=True, axis_name='batch'
    )
    dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
    # dynamic loss takes care of averaging gradients across replicas
  else:
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    aux, grads = grad_fn(state.params)
    # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
    grads = lax.pmean(grads, axis_name='batch')
  new_model_state, logits = aux[1]
  metrics = compute_metrics(logits, batch['label'])
  metrics['lr'] = lr

  new_state = state.apply_gradients(
    grads=grads, batch_stats=new_model_state['batch_stats']
  )
  if dynamic_scale:
    # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
    # params should be restored (= skip this step).
    new_state = new_state.replace(
      opt_state=jax.tree_util.tree_map(
        functools.partial(jnp.where, is_fin),
        new_state.opt_state,
        state.opt_state,
      ),
      params=jax.tree_util.tree_map(
        functools.partial(jnp.where, is_fin), new_state.params, state.params
      ),
      dynamic_scale=dynamic_scale,
    )
    metrics['scale'] = dynamic_scale.scale

  return new_state, metrics


def eval_step(state, batch):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  logits = state.apply_fn(variables, batch['image'], train=False, mutable=False, rng=jax.random.PRNGKey(0))
  return compute_metrics(logits, batch['label'])


class TrainState(train_state.TrainState):
  batch_stats: Any
  dynamic_scale: dynamic_scale_lib.DynamicScale


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


def sync_batch_stats(state):
  """Sync the batch statistics across replicas."""
  # Each device has its own version of the running average batch statistics and
  # we sync them before evaluation.
  if hasattr(state, 'batch_stats'):
    return state
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def create_train_state(
    rng, config: ml_collections.ConfigDict, model, image_size, learning_rate_fn
):
  """
  Create initial training state, including the model and optimizer.
  """
  # print("here we are in the function 'create_train_state' in train.py; ready to define optimizer")

  dynamic_scale = None
  platform = jax.local_devices()[0].platform
  if config.half_precision and platform == 'gpu':
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  else:
    dynamic_scale = None

  params, batch_stats = initialized(rng, image_size, model)
  
  print_params(params)

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
    tx = optax.adamw(
      learning_rate=learning_rate_fn,
      b1=0.9,
      b2=0.999,
      eps=1e-8,
      weight_decay=config.weight_decay,
      # grad_norm_clip=grad_norm_clip, # None if no clipping
    )
  else:
    raise ValueError(f'Unknown optimizer: {config.optimizer}, choose from "sgd" or "adamw"')
  state = TrainState.create(
    apply_fn=model.apply,
    params=params,
    tx=tx,
    batch_stats=batch_stats,
    dynamic_scale=dynamic_scale,
  )
  return state


def train_and_evaluate(
    config: ml_collections.ConfigDict, workdir: str
) -> TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.
    workdir: Directory where the tensorboard summaries are written to.

  Returns:
    Final TrainState.
  """

  writer = metric_writers.create_default_writer(
      logdir=workdir, just_logging=jax.process_index() != 0
  )

  rng = random.key(config.seed)

  image_size = 224

  logging.info('config.batch_size: {}'.format(config.batch_size))

  # print("position 1")

  if config.batch_size % jax.process_count() > 0:
    raise ValueError('Batch size must be divisible by the number of processes')
  local_batch_size = config.batch_size // jax.process_count()
  logging.info('local_batch_size: {}'.format(local_batch_size))
  logging.info('jax.local_device_count: {}'.format(jax.local_device_count()))

  if local_batch_size % jax.local_device_count() > 0:
    raise ValueError('Local batch size must be divisible by the number of local devices')

  train_loader, steps_per_epoch = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='train',
    # split='val' if config.debug else 'train',
  )
  eval_loader, steps_per_eval = input_pipeline.create_split(
    config.dataset,
    local_batch_size,
    split='val',
  )
  # print("position 2")
  logging.info('steps_per_epoch: {}'.format(steps_per_epoch))
  logging.info('steps_per_eval: {}'.format(steps_per_eval))

  if config.steps_per_eval != -1:
    steps_per_eval = config.steps_per_eval

  base_learning_rate = config.learning_rate * config.batch_size / 512.0 # note that here the input config.learning_rate is 0.0005 in the paper

  model_cls = getattr(models, config.model)
  model = create_model(
    model_cls=model_cls, half_precision=config.half_precision,
    dropout_rate=config.dropout_rate,
    stochastic_depth_rate=config.stochastic_depth_rate,
  )

  learning_rate_fn = create_learning_rate_fn(config, base_learning_rate, steps_per_epoch)

  # print("position 3")

  state = create_train_state(rng, config, model, image_size, learning_rate_fn)
  state = restore_checkpoint(state, workdir)
  # step_offset > 0 if restarting from checkpoint
  step_offset = int(state.step)
  epoch_offset = step_offset // steps_per_epoch  # sanity check for resuming
  assert epoch_offset * steps_per_epoch == step_offset, (epoch_offset, steps_per_epoch, step_offset)
  state = jax_utils.replicate(state)
  # reload checkpoint done

  # use pmap to parallel training
  # p_train_step = jax.pmap(
  #     functools.partial(train_step, rng_init=rng, learning_rate_fn=learning_rate_fn),
  #     axis_name='batch',
  # )
  p_train_step = jax.pmap(
    functools.partial(train_step_sqa, rng_init=rng, learning_rate_fn=learning_rate_fn, label_smoothing=0.0),
    axis_name='batch',
  )
  p_eval_step = jax.pmap(eval_step, axis_name='batch')

  train_metrics = []
  hooks = []
  # if jax.process_index() == 0:
  #   hooks += [periodic_actions.Profile(num_profile_steps=5, logdir=workdir)]
  train_metrics_last_t = time.time()
  logging.info('Initial compilation, this might take some minutes...')
  for epoch in range(epoch_offset, config.num_epochs):
    if jax.process_count() > 1:
      train_loader.sampler.set_epoch(epoch)
    logging.info('epoch {}...'.format(epoch))
    # print("train_loader: " , train_loader)
    # print("train_loader.sampler: ", train_loader.sampler)
    # print("length of train_loader: ", len(train_loader))
    # print("length of train_loader.sampler: ", len(train_loader.sampler))
    # print("length of train_loader.dataset: ", len(train_loader.dataset))
    for n_batch, batch in enumerate(train_loader):
      # print("number of batch:", n_batch)
      # print("shape of the batch images:", batch[0].shape)
      batch = pre_process_batch(batch)
      batch = apply_mixup_cutmix_batch(config.dataset, batch)
      step = epoch * steps_per_epoch + n_batch
      # print(batch[0].shape)
      batch = prepare_batch_data_sqa(batch)

      # print("batch['image'].shape:", batch['image'].shape)
      # print("batch['label'].shape:", batch['label'].shape)
      # assert False

      
      # # here is code for us to visualize the images
      # import matplotlib.pyplot as plt
      # import numpy as np
      # import os
      # print(batch["image"].shape)
      # # print(batch["label"].shape)

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
      # assert batch['label'].shape == (1, local_batch_size, 1000) # the first dimension is the number of devices
      assert batch['label'].shape[-1] == NUM_CLASSES
      state, metrics = p_train_step(state, batch) # here is the training step
      
      if epoch == epoch_offset and n_batch == 0:
        logging.info('Initial compilation completed. Reset timer.')
        train_metrics_last_t = time.time()
      
      for h in hooks:
        h(step)

      # normalize to IN1K epoch anyway
      ep = step * config.batch_size / 1281167

      if config.get('log_per_step'):
        train_metrics.append(metrics)
        if (step + 1) % config.log_per_step == 0:
          # print('Hello')
          train_metrics = common_utils.get_metrics(train_metrics)
          train_metrics.pop('labels')  # used in val only
          summary = {
            f'train_{k}': v
            for k, v in jax.tree_util.tree_map(
                lambda x: float(x.mean()), train_metrics
            ).items()
          }
          summary['steps_per_second'] = config.log_per_step / (time.time() - train_metrics_last_t)
          # summary['seconds_per_step'] = (time.time() - train_metrics_last_t) / config.log_per_step

          # step for tensorboard
          summary["ep"] = ep

          writer.write_scalars(step + 1, summary)
          train_metrics = []
          train_metrics_last_t = time.time()

    # logging per epoch
    if (epoch + 1) % config.eval_per_epoch == 0:
      logging.info('Eval epoch {}...'.format(epoch))
      eval_metrics = []
      # sync batch statistics across replicas
      state = sync_batch_stats(state)
      for n_eval_batch, eval_batch in enumerate(eval_loader):
        if (n_eval_batch + 1) % config.log_per_step == 0:
          logging.info('eval: {}/{}'.format(n_eval_batch + 1, steps_per_eval))
        eval_batch = prepare_batch_data_sqa(eval_batch, local_batch_size)

        metrics = p_eval_step(state, eval_batch) # here is the eval step
        # print("metrics' labels shape:", metrics['labels'].shape)
        assert metrics['labels'].shape[-1] == NUM_CLASSES
        eval_metrics.append(metrics)

      eval_metrics = common_utils.get_metrics(eval_metrics) # loss, acc, labels
      eval_metrics_copy = eval_metrics # labels shape: (local_batch_size, 1000)
      eval_metrics = jax.tree_map(lambda x: x.flatten(), eval_metrics)
      logging.info('evaluated samples: {}'.format(eval_metrics['labels'].size))
      valid = (eval_metrics_copy['labels'] >= 0)
      # print(valid.shape)
      # print(eval_metrics_copy['labels'].shape)
      # print(eval_metrics_copy)
      # print(eval_metrics["labels"].shape)
      # print(eval_metrics)
      # assert valid.shape[-1] == NUM_CLASSES

      # print(valid.shape)
      # for key, val in eval_metrics_copy.items():
      #   print(key, val.shape)
      # for key, val in eval_metrics.items():
      #   print(key, val.shape)
      
      # valid shape: 
      # print(valid.shape)
      valid = valid.reshape(-1, NUM_CLASSES)
      valid = valid[:, 0] # only take the first column, because we only need to pick out these valid samples
      # print(valid.shape)
      assert valid.ndim == 1
      # omit label in eval_metrics
      eval_metrics = {
        'loss': eval_metrics['loss'],
        'accuracy': eval_metrics['accuracy'],
      }
      eval_metrics = jax.tree_map(lambda x: x[valid], eval_metrics)
      logging.info('valid samples: {}'.format(eval_metrics['loss'].size))

      summary = jax.tree_util.tree_map(lambda x: float(x.mean()), eval_metrics)
      logging.info(
        'eval epoch: %d, loss: %.6f, accuracy: %.6f',
        epoch,
        summary['loss'],
        summary['accuracy'] * 100,
      )
      summary = {f'eval_{key}': val for key, val in summary.items()}
      summary["ep"] = ep
      writer.write_scalars(step + 1, summary)
      writer.flush()

    if (
      (epoch + 1) % config.checkpoint_per_epoch == 0
      or epoch == config.num_epochs
      or epoch == 0  # saving at the first epoch for sanity check
    ):
      state = sync_batch_stats(state)
      # TODO{km}: suppress the annoying warning.
      save_checkpoint(state, workdir)

  # Wait until computations are done before exiting
  jax.random.normal(jax.random.key(0), ()).block_until_ready()

  return state