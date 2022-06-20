"""Flow implementing "Distributionally Robust Neural Network For Group 
Shifts: On the Importance of Regularization for Worst-Case Generalization". 
See https://arxiv.org/pdf/1911.08731.pdf.
"""
import os
import torch
import random 
import numpy as np
from pprint import pprint
from os.path import join
from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.systems import ReviewDataModule, RobustSentimentSystem
from src.paths import LOG_DIR, CONFIG_DIR
from src.utils import load_config, to_json


class DistRobustOpt(FlowSpec):
  r"""A flow that implements Equation 4 on page 3 of the paper. 

  We assume access to group labels, meaning whether an example is in 
  English or Spanish (this should be quite easy to obtain). We do not 
  assume access to this in test sets. Then, we minimize the maximum 
  group loss over all group.
  """
  config_path = Parameter('config', 
    help = 'path to config file', default = join(CONFIG_DIR, 'dro.json'))
  
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
    dm = ReviewDataModule(config)

    # your implementation will be used here!
    system = RobustSentimentSystem(config)

    checkpoint_callback = ModelCheckpoint(
      dirpath = config.system.save_dir,
      save_last = True,  # save the last epoch!
      verbose = True,
    )

    trainer = Trainer(
      logger = TensorBoardLogger(save_dir=LOG_DIR),
      max_epochs = config.system.optimizer.max_epochs,
      callbacks = [checkpoint_callback])

    self.dm = dm
    self.system = system
    self.trainer = trainer

    self.next(self.train_dro)

  @step
  def train_dro(self):
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

    pprint(results)  # print results to command line

    log_file = join(LOG_DIR, 'dro_flow', 'results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python dro_flow.py`. To list
  this flow, run `python dro_flow.py show`. To execute
  this flow, run `python dro_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python dro_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python dro_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = DistRobustOpt()
