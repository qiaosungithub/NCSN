# Copied from Kaiming He's resnet_jax repository

import os
from absl import app
from absl import flags
from absl import logging
from clu import platform
import jax
from ml_collections import config_flags

import train_sqa
from utils.utils import logging_util

import warnings
warnings.filterwarnings("ignore")

FLAGS = flags.FLAGS

# define input parameters

flags.DEFINE_string('workdir', None, 'Directory to store model data.')
flags.DEFINE_bool('debug', False, 'Debugging mode.')
config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)

def main(argv):
  # print("argv: ",argv) # main.py
  # print("flags: ",FLAGS) # see flags.md
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # 记录JAX进程信息
  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # Add a note so that we can tell which task is which JAX host.
  # (Depending on the platform task 0 is not guaranteed to be host 0)
  platform.work_unit().set_task_status(
      f'process_index: {jax.process_index()}, '
      f'process_count: {jax.process_count()}'
  )
  platform.work_unit().create_artifact(
      platform.ArtifactType.DIRECTORY, FLAGS.workdir, 'workdir'
  )
  # print("FLAGS.config: ",FLAGS.config) # also see flags.md

  if FLAGS.debug:
    with jax.disable_jit():
      train_sqa.train_and_evaluate(FLAGS.config, FLAGS.workdir)
  else:
    train_sqa.train_and_evaluate(FLAGS.config, FLAGS.workdir)


if __name__ == '__main__':
  logging_util.verbose_off()
  logging_util.set_time_logging(logging)
  flags.mark_flags_as_required(['config', 'workdir']) # use flags to parse the input parameters
  app.run(main)
