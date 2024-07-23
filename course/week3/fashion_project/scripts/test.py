# Do not edit this file.

import os
import torch
import random
import numpy as np
from os.path import join
from pprint import pprint
from torch.utils.data import DataLoader

from pytorch_lightning import Trainer
from metaflow import FlowSpec, step, Parameter
from fashion.system import FashionDataModule, FashionClassifierSystem
from fashion.system import ProductionDataset
from fashion.utils import to_json
from fashion.paths import CONFIG_DIR, LOG_DIR, CHECKPOINT_DIR, DATA_DIR


class TestFlow(FlowSpec):
  r"""A MetaFlow that evaluates a image classifier to recognize images of fashion clothing
  on production data.

  Arguments
  ---------
  config (str, default: ./configs/test.json): path to a configuration file
  test (str, default: offline)
  checkpoint (str, default: ./checkpoints/model.ckpt)
  """
  test_type = Parameter('test', help='test type to run', default = 'production') # offline
  checkpoint_path = Parameter('checkpoint', help = 'path to checkpoint file', default = join(CHECKPOINT_DIR, 'model.ckpt'))

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.test)

  @step
  def test(self):
    r"""Performs evaluation on a trained model.
    Depends on the `self.test_type` parameter. 
    - If `offline`, then computes accuracy on the fixed FashionMNIST test set
    - If `production`, then computes accuracy on set of production data
    """
    trainer = Trainer()

    # Load trained system
    system = FashionClassifierSystem.load_from_checkpoint(self.checkpoint_path)

    if self.test == "offline":
      dm = FashionDataModule()
      trainer.test(system, dm, ckpt_path = self.checkpoint_path)
      results = system.test_results
      log_name = 'offline.json'
    else:
      # We pretend we have access to all the labels to compute these results
      # In reality, we do not have these hidden labels accessible.
      ds = ProductionDataset(join(DATA_DIR, 'production/dataset.pt'), return_hidden_labels = True)
      dl = DataLoader(ds, batch_size=10)
      trainer.test(system, dataloaders = dl, ckpt_path = self.checkpoint_path)
      results = system.test_results
      log_name = 'production.json'

    # print results to command line
    pprint(results)

    # save to disk
    log_file = join(LOG_DIR, log_name)
    os.makedirs(LOG_DIR, exist_ok = True)
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

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python test.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python test.py resume`
  
  You can specify a run id as well.
  """
  flow = TestFlow()
