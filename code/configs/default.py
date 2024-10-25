# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2021 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Model
  config.model = 'NCSNv2'

  # Dataset
  config.dataset = dataset = ml_collections.ConfigDict()
  dataset.name = 'MNIST'
  dataset.image_size = 28
  dataset.channels = 1
  dataset.root = '/kmh-nfs-ssd-eu-mount/code/qiao/data/MNIST/'
  dataset.num_workers = 4
  dataset.prefetch_factor = 2
  dataset.pin_memory = False
  dataset.cache = False

  # Training
  config.learning_rate = 0.1
  config.momentum = 0.9
  config.batch_size = 128
  config.eval_batch_size = 500
  config.shuffle_buffer_size = 16 * 128
  config.prefetch = 10

  config.num_epochs = 100
  config.log_per_step = 100
  config.log_per_epoch = -1
  config.eval_per_epoch = 1
  config.checkpoint_per_epoch = 20

  config.steps_per_eval = -1
  
  config.half_precision = False

  config.seed = 0  # init random seed
  config.spec_norm = False
  config.normalization = "InstanceNorm++"
  config.activation = "elu"
  config.ngf = 64

  # sampling
  config.sigma_begin = 28
  config.n_noise_levels = 75
  config.sigma_end = 0.01
  config.ema = True
  config.ema_decay = 0.999
  
  
  return config


def metrics():
  return [
    'train_loss',
    'eval_loss',
    'train_accuracy',
    'eval_accuracy',
    'steps_per_second',
    'train_learning_rate',
  ]
