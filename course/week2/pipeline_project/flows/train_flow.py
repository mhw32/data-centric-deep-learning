"""
Flow #1: This flow will train a small (linear) neural network 
on the MNIST dataset to performance classification.
"""

import os
import wandb
import torch
import random
import shutil
import numpy as np
from os.path import join
from pathlib import Path
from pprint import pprint

from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.system import MNISTDataModule, DigitClassifierSystem
from src.utils import load_config, to_json


class DigitClassifierFlow(FlowSpec):
  r"""A flow that trains a image classifier to recognize handwritten
  digit, such as those in the MNIST dataset.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default='./configs/train_flow.json')

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # uncomment me when logging
    # wandb.init()  # initialize wandb module

    self.next(self.init_system)

  @step
  def init_system(self):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance.
    """
    # configuration files contain all hyperparameters
    config = load_config(self.config_path)

    # a data module wraps around training, dev, and test datasets
    dm = MNISTDataModule(config)

    # a PyTorch Lightning system wraps around model logic
    system = DigitClassifierSystem(config)

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = config.system.save_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    # Logging is an important part of training a model. It helps us understand
    # what the model is doing and look out for early signs that something might 
    # be going wrong. We will be using 'Weights and Biases', a relatively new 
    # tool that makes logging in the cloud easy. 
    # 
    # wandb_logger = WandbLogger(
    #   project = config.wandb.project, 
    #   offline = False,
    #   entity = config.wandb.entity, 
    #   name = 'mnist', 
    #   save_dir = 'logs/wandb',
    #   config = config)

    trainer = Trainer(
      max_epochs = config.system.optimizer.max_epochs,
      # logger = wandb_logger,
      callbacks = [checkpoint_callback])

    # when we save these objects to a `step`, they will be available
    # for use in the next step, through not steps after.
    self.dm = dm
    self.system = system
    self.trainer = trainer

    self.next(self.train_model)

  @step
  def train_model(self):
    """Calls `fit` on the trainer."""

    # Call `fit` on the trainer with `system` and `dm`.
    # Our solution is one line.
    self.trainer.fit(self.system, self.dm)

    # uncomment me when logging
    # wandb.finish()  # close wandb run

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
      f'logs/train_flow', 'offline-test-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python train_flow.py`. To list
  this flow, run `python train_flow.py show`. To execute
  this flow, run `python train_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python train_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python train_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = DigitClassifierFlow()
