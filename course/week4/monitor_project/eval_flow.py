"""Suppose we have collected test sets for both groups: english and 
spanish for the purpose of evaluation. 
"""

import os
import torch
import random
import numpy as np
from pprint import pprint
from os.path import join
from torch.utils.data import DataLoader
from metaflow import FlowSpec, step, Parameter

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.systems import SentimentClassifierSystem
from src.dataset import ProductReviewEmbeddings
from src.paths import LOG_DIR, CONFIG_DIR
from src.utils import load_config, to_json


class EvalClassifier(FlowSpec):
  r"""A flow that evaluates a trained sentiment classifier on sets
  of English and Spanish reviews. In the data distribution, these two
  groups are not evenly balanced. This flow serves as an evaluation 
  for how well the model does on each group individually.

  Arguments
  ---------
  config (str, default: ./configs/eval.json): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default = join(CONFIG_DIR, 'eval.json'))
  
  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.load_system)

  @step
  def load_system(self):
    r"""Load pretrain system on new training data."""
    config = load_config(self.config_path)
    system = SentimentClassifierSystem.load_from_checkpoint(config.system.ckpt_path)
    trainer = Trainer(logger = TensorBoardLogger(save_dir=LOG_DIR))

    self.system = system
    self.trainer = trainer
  
    self.next(self.evaluate)

  @step
  def evaluate(self):
    r"""Evaluate system on two different test datasets."""

    config = load_config(self.config_path)
    en_ds = ProductReviewEmbeddings(lang='en', split='test')
    es_ds = ProductReviewEmbeddings(lang='es', split='test')

    en_dl = DataLoader(en_ds, batch_size = config.system.batch_size, 
      num_workers = config.system.num_workers)
    es_dl = DataLoader(es_ds, batch_size = config.system.batch_size, 
      num_workers = config.system.num_workers)

    self.trainer.test(self.system, dataloaders = en_dl)
    en_results = self.system.test_results

    self.trainer.test(self.system, dataloaders = es_dl)
    es_results = self.system.test_results

    print('Results on English reviews:')
    pprint(en_results)

    print('Results on Spanish reviews:')
    pprint(es_results)

    log_file = join(LOG_DIR, 'eval_flow', 'en_results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(en_results, log_file)  # save to disk

    log_file = join(LOG_DIR, 'eval_flow', 'es_results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(es_results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python eval_flow.py`. To list
  this flow, run `python eval_flow.py show`. To execute
  this flow, run `python eval_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python eval_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python eval_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = EvalClassifier()
