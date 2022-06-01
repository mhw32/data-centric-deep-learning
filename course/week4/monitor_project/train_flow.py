"""This flow will train a neural network to perform sentiment classification 
for Amazon reviews across several product categories.
"""

import os
import torch
import random
import numpy as np
from os.path import join
from pprint import pprint

from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.systems import ReviewDataModule, SentimentClassifierSystem
from src.paths import LOG_DIR, CONFIG_DIR
from src.utils import load_config, to_json


class TrainClassifier(FlowSpec):
  r"""A flow that trains a natural language inference model.

  Arguments
  ---------
  config (str, default: ./configs/train.json): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default = join(CONFIG_DIR, 'train.json'))

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
    # configuration files contain all hyperparameters
    config = load_config(self.config_path)

    # a data module wraps around training, dev, and test datasets
    dm = ReviewDataModule(config)

    # a PyTorch Lightning system wraps around model logic
    system = SentimentClassifierSystem(config)

    # a callback to save best model weights
    checkpoint_callback = ModelCheckpoint(
      dirpath = config.system.save_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    trainer = Trainer(
      logger = TensorBoardLogger(save_dir=LOG_DIR),
      max_epochs = config.system.optimizer.max_epochs,
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

    self.next(self.offline_test)

  @step
  def offline_test(self):
    r"""Calls (offline) `test` on the trainer. Saves results to a log file."""

    # Load the best checkpoint and compute results using `self.trainer.test`
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')
    results = self.system.test_results

    # print results to command line
    pprint(results)

    log_file = join(LOG_DIR, 'train_flow', 'results.json')
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
  flow = TrainClassifier()
