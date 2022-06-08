"""
Flow #6: This flow will train a resnet18 network
on the MNIST dataset to performance classification, and run an
integration test to measure model performance in-the-wild.
"""

import os
import torch
import random
import numpy as np
import wandb
from os.path import join
from pathlib import Path
from pprint import pprint

from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from src.system import MNISTDataModule, DigitClassifierSystem
from src.tests.integration import MNISTIntegrationTest
from src.tests.regression import MNISTRegressionTest
from src.tests.directionality import MNISTDirectionalityTest
from src.utils import load_config, to_json


class DigitClassifierFlow(FlowSpec):
    r"""A flow that trains a image classifier to recognize handwritten
    digit, such as those in the MNIST dataset. Includes an integration
    test on handwritten digits provided by you!

    Arguments
    ---------
    config (str, default: ./config.py): path to a configuration file
    """
    config_path = Parameter('config',
      help = 'path to config file', default='./configs/resnet18_flow.json')

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

      dm = MNISTDataModule(config)
      system = DigitClassifierSystem(config)

      checkpoint_callback = ModelCheckpoint(
            dirpath = config.system.save_dir,
            monitor = 'dev_loss',
            mode = 'min',    # look for lowest `dev_loss`
            save_top_k = 1,  # save top 1 checkpoints
            verbose = True,
      )

      trainer = Trainer(
            max_epochs = config.system.optimizer.max_epochs,
            callbacks = [checkpoint_callback],
            accelerator='mps',
      )

      self.dm = dm
      self.system = system
      self.trainer = trainer
      self.config = config

      self.next(self.train_model)

    @step
    def train_model(self):
      """Calls `fit` on the trainer."""

      with wandb.init(project=self.config.wandb.project):
        wandb_logger = WandbLogger(
          project = self.config.wandb.project,
          offline = False,
          entity = self.config.wandb.entity,
          name = 'mnist_resnet18',
          save_dir = 'logs/wandb',
          config = self.config)
        self.trainer.logger = wandb_logger

        self.trainer.fit(self.system, self.dm)

      self.trainer.logger = None

      self.next(self.offline_test, self.integration_test, self.regression_test, self.directionality_test)

    @step
    def offline_test(self):
      r"""Calls (offline) `test` on the trainer. Saves results to a log file."""

      # Load the best checkpoint and compute results using `self.trainer.test`
      self.trainer.test(self.system, self.dm, ckpt_path = 'best')
      results = self.system.test_results

      # print results to command line
      pprint(results)

      log_file = join(Path(__file__).resolve().parent.parent,
            f'logs/resnet18_flow', 'offline-test-results.json')

      os.makedirs(os.path.dirname(log_file), exist_ok = True)
      to_json(results, log_file)  # save to disk

      self.next(self.join)

    @step
    def integration_test(self):
      r"""Runs an integration test. Saves results to a log file."""

      test = MNISTIntegrationTest()
      test.test(self.trainer, self.system)

      results = self.system.test_results
      pprint(results)

      log_file = join(Path(__file__).resolve().parent.parent,
            f'logs/resnet18_flow', 'integration-test-results.json')

      os.makedirs(os.path.dirname(log_file), exist_ok = True)
      to_json(results, log_file)

      self.next(self.join)

    @step
    def regression_test(self):
        test = MNISTRegressionTest()
        test.test(self.trainer, self.system)

        results = self.system.test_results
        pprint(results)

        log_file = join(Path(__file__).resolve().parent.parent,
          f'logs/regression_flow', 'regression-test-results.json')

        os.makedirs(os.path.dirname(log_file), exist_ok = True)
        to_json(results, log_file)

        self.next(self.join)

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

        self.next(self.join)

    @step
    def join(self, inputs):
        self.next(self.end)

    @step
    def end(self):
      """End node!"""
      print('done! great work!')


if __name__ == "__main__":
    """
    To validate this flow, run `python resnet18_flow.py`. To list
    this flow, run `python resnet18_flow.py show`. To execute
    this flow, run `python resnet18_flow.py run`.

    You may get PyLint errors from `numpy.random`. If so,
    try adding the flag:

      `python resnet18_flow.py --no-pylint run`

    If you face a bug and the flow fails, you can continue
    the flow at the point of failure:

      `python resnet18_flow.py resume`

    You can specify a run id as well.
    """
    flow = DigitClassifierFlow()
