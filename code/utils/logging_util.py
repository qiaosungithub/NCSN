# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging as _logging
from absl import logging

import jax
from jax.experimental import multihost_utils

from termcolor import colored
import time


def log_for_0(*args):
    if jax.process_index() == 0:
        logging.info(*args)

class Timer:
    def __init__(self):
        self.start_time = time.time()

    def elapse_without_reset(self):
        return time.time() - self.start_time

    def elapse_with_reset(self):
        """This do both elaspse and reset"""
        a = time.time() - self.start_time
        self.reset()
        return a

    def reset(self):
        self.start_time = time.time()

    def __str__(self):
        return f'{self.elapse_with_reset():.2f} s'