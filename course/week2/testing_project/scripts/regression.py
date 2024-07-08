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
from testing.regression import MNISTRegressionTest
from testing.utils import load_config, to_json
from testing.paths import CONFIG_DIR, LOG_DIR


class RegressionTest(FlowSpec):
  r"""A flow that trains a image classifier to recognize handwritten
  digit, such as those in the MNIST dataset. Includes an regression 
  test using a trained model.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', help='path to config file', default = join(CONFIG_DIR, 'test.json'))

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
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    config = load_config(self.config_path)

    self.dm = MNISTDataModule(config)
    self.system = DigitClassifierSystem.load_from_checkpoint(config.model)

    self.next(self.offline_test)

  @step
  def offline_test(self):
    r"""Calls (offline) `test` on the trainer. Saves results to a log file."""

    # Load the best checkpoint and compute results using `self.trainer.test`
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')
    results = self.system.test_results

    # print results to command line
    pprint(results)

    log_file = join(LOG_DIR, 'offline-test-results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.regression_test)

  @step
  def regression_test(self):
    r"""Runs an integration test. Saves results to a log file."""

    test = MNISTRegressionTest()
    test.test(self.trainer, self.system)

    results = self.system.test_results
    pprint(results)

    log_file = join(LOG_DIR, 'regression-test-results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)

    self.results = results
    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python regression_flow.py`. To list
  this flow, run `python regression_flow.py show`. To execute
  this flow, run `python regression_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python regression_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python regression_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = RegressionTest()
