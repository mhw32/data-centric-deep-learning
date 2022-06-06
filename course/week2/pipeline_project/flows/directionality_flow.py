"""
Flow #3: This flow will train a small (linear) neural network 
on the MNIST dataset to performance classification, and run an 
directionality test to measure model robustness.
"""

import os
import wandb
import torch
import random
import numpy as np
from os.path import join
from pathlib import Path
from pprint import pprint

from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.system import MNISTDataModule, DigitClassifierSystem
from src.tests.directionality import MNISTDirectionalityTest
from src.utils import load_config, to_json


class DigitClassifierFlow(FlowSpec):
  r"""A flow that trains a image classifier to recognize handwritten
  digit, such as those in the MNIST dataset. Includes a directionality 
  test on transformed digits to test model robustness.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default='./configs/directionality_flow.json')

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    wandb.init()  # start wandb run

    self.next(self.init_system)

  @step
  def init_system(self):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    config = load_config(self.config_path)

    dm = MNISTDataModule(config)
    system = DigitClassifierSystem(config)

    checkpoint_callback = ModelCheckpoint(
      dirpath = config.system.save_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    wandb_logger = WandbLogger(
      project = config.wandb.project, 
      offline = False,
      entity = config.wandb.entity, 
      name = 'mnist', 
      save_dir = 'logs/wandb',
      config = config)

    trainer = Trainer(
      max_epochs = config.system.optimizer.max_epochs,
      logger = wandb_logger,
      callbacks = [checkpoint_callback])

    self.dm = dm
    self.system = system
    self.trainer = trainer

    self.next(self.train_model)

  @step
  def train_model(self):
    """Calls `fit` on the trainer."""

    self.trainer.fit(self.system, self.dm)

    wandb.finish()  # close wandb run

    self.next(self.offline_test)

  @step
  def offline_test(self):
    r"""Calls (offline) `test` on the trainer. Saves results to a log file."""

    # Load the best checkpoint and compute results using `self.trainer.test`
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')
    results = self.system.test_results

    # print results to command line
    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs/directionality_flow', 'offline-test-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.directionality_test)

  @step
  def directionality_test(self):
    r"""Runs an directionality test. Saves results to a log file."""

    test = MNISTDirectionalityTest()
    test.test(self.trainer, self.system)

    results = self.system.test_results
    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs/directionality_flow', 'directionality-test-results.json')

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
  To validate this flow, run `python directionality_flow.py`. To list
  this flow, run `python directionality_flow.py show`. To execute
  this flow, run `python directionality_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python directionality_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python directionality_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = DigitClassifierFlow()
