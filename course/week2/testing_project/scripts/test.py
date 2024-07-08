"""
Flow #3: This flow will train a small (linear) neural network 
on the MNIST dataset to performance classification, and run an 
regression test to measure when a model is struggling.
"""

import os
import torch
import random
import numpy as np
from os.path import join
from pathlib import Path
from pprint import pprint

from metaflow import FlowSpec, step, Parameter
from testing.system import MNISTDataModule, DigitClassifierSystem
from testing.integration import MNISTIntegrationTest
from testing.regression import MNISTRegressionTest
from testing.directionality import MNISTDirectionalityTest
from testing.utils import load_config, to_json
from testing.paths import CONFIG_DIR, LOG_DIR, IMAGE_DIR


class TestFlow(FlowSpec):
  r"""A flow that trains a image classifier to recognize handwritten
  digit, such as those in the MNIST dataset. Includes an regression 
  test using a trained model.

  Arguments
  ---------
  config_path (str, default: ./config.py): path to a configuration file
  test_type (str, default: offline): offline | integration | regression | directionality
  """
  config_path = Parameter(
    'config', 
    help='path to config file', 
    default = join(CONFIG_DIR, 'test.json'), 
    required = True,
  )
  test_type = Parameter(
    'test', 
    help='test type to run', 
    default = 'offline', 
    choices = ['offline', 'integration', 'regression', 'directionality'], 
    required = True,
  )
  image_dir = Parameter(
    'image_dir',
    help='path to evaluation images',
    default = IMAGE_DIR,
    required = True,
  )

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.init_system)

  @step
  def init_system(self):
    r"""Loads a trained deep learning model.
    """
    config = load_config(self.config_path)

    self.dm = MNISTDataModule(config)
    self.system = DigitClassifierSystem.load_from_checkpoint(config.model)

    self.next(self.offline_test)

  @step
  def test(self):
    r"""Performs evaluation on a trained model.
    Depends on the `self.test_type` parameter. 
    - If `offline`, then computes accuracy on the fixed MNIST test set
    - If `integration`, then computes accuracy on your handwritten digits
    - If `regression`, then computes accuracy comparing a linear and a mlp 
    - If `directionality`, then compute agreement between a perturbed & a non-perturbed image
    """
    if self.test_type == "offline":
      # Load the best checkpoint and compute results using `self.trainer.test`
      self.trainer.test(self.system, self.dm, ckpt_path = 'best')
      results = self.system.test_results
      log_name = 'offline-test-results.json'
    elif self.test_type == "integration":
      test_dir = join(self.image_dir, "integration")
      MNISTIntegrationTest(test_dir).test(self.trainer, self.system)
      results = self.system.test_results
      log_name = 'integration-test-results.json'
    elif self.test_type == "regression":
      test_dir = join(self.image_dir, "regression")
      MNISTRegressionTest(test_dir).test(self.trainer, self.system)
      results = self.system.test_results
      log_name = 'regression-test-results.json'
    elif self.test_type == "directionality":
      # directionality test uses integration examples
      test_dir = join(self.image_dir, "integration")
      MNISTDirectionalityTest(test_dir).test(self.trainer, self.system)
      results = self.system.test_results
      log_name = 'directionality-test-results.json'
    else:
      raise Exception(f'Test type {self.test_type} is not supported.')

    # print results to command line
    pprint(results)

    # save to disk
    log_file = join(LOG_DIR, log_name)
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python test.py`. To list
  this flow, run `python test.py show`. To execute
  this flow, run `python test.py run`.

  To set the test type, use the `--test` flag. For example,

    `python test.py run --test integration`

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python test.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python test.py resume`
  
  You can specify a run id as well.
  """
  flow = TestFlow()
