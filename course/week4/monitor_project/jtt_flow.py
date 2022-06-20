"""Flow implementing "Just Train Twice: Improving Group Robustness without 
Training Group Information". See https://arxiv.org/pdf/2107.09044.pdf.
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
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.systems import ReviewDataModule, SentimentClassifierSystem
from src.dataset import ProductReviewEmbeddings
from src.paths import LOG_DIR, CONFIG_DIR
from src.utils import load_config, to_json


class JustTrainTwice(FlowSpec):
  r"""A flow that implements Algorithm 1 on page 5 of the paper. 

  We build a dataset E of training examples misclassified by the 
  trained model. Then, we retrain the model from scratch and upweight
  the samples in E. We choose between multiple upweight parameters 
  to see which one results in the best performance.
  """
  config_path = Parameter('config', 
    help = 'path to config file', default = join(CONFIG_DIR, 'jtt.json'))
  
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
    system = SentimentClassifierSystem.load_from_checkpoint(
      config.system.pretrain.ckpt_path)
    trainer = Trainer(logger = TensorBoardLogger(save_dir=LOG_DIR))

    self.config = config
    self.system = system
    self.trainer = trainer
  
    self.next(self.build_weights)

  @step
  def build_weights(self):
    r"""Build map from example to weight."""

    ds = ProductReviewEmbeddings(lang='mix', split='train')
    dl = DataLoader(ds, batch_size=self.config.system.optimizer.batch_size, 
      num_workers=self.config.system.optimizer.num_workers)

    weights = None
    # =============================
    # FILL ME OUT
    # 
    # Find out which examples in the training dataset the trained model gets 
    # incorrect. We expect the variable `weights` to be a `torch.FloatTensor`
    # of the same size as `len(ds)`. The entries in `weights` are either 0 or 
    # 1 where the entry is 1 if the model is incorrect. 
    # 
    # Pseudocode:
    # --
    # Get predicted probabilities with `self.trainer` on the DataLoader `dl`.
    # Round probabilities to predictions, and compare to labels. 
    # Element-wise comparison from predictions to labels to see if 
    #   each element matches. 
    # Store the result into `weights`.
    # 
    # Type:
    # --
    # weights: torch.FloatTensor (length: |ds|)
    # =============================
    self.weights = weights
    
    # search through all of these lambda for upweighting 
    self.lambd = [5, 10, 20 ,30, 40, 50, 100]
    self.next(self.retrain, foreach='lambd')

  @step
  def retrain(self):
    lambd = self.input
    config = self.config

    # add weights to training set
    dm = ReviewDataModule(config, weights = self.weights)
    system = SentimentClassifierSystem(config)

    checkpoint_callback = ModelCheckpoint(
      # save to its own folder
      dirpath = join(config.system.save_dir, f'lambd_{lambd}'),
      monitor = 'dev_loss',
      mode = 'min',
      save_top_k = 1,
      verbose = True,
    )

    trainer = Trainer(
      logger = TensorBoardLogger(save_dir = LOG_DIR),
      max_epochs = config.system.optimizer.max_epochs,
      callbacks = [checkpoint_callback])

    # train model
    trainer.fit(system, dm)

    en_ds = ProductReviewEmbeddings(lang='en', split='test')
    es_ds = ProductReviewEmbeddings(lang='es', split='test')
    en_dl = DataLoader(en_ds, 
      batch_size = config.system.optimizer.batch_size, 
      num_workers = config.system.optimizer.num_workers)
    es_dl = DataLoader(es_ds, 
      batch_size = config.system.optimizer.batch_size, 
      num_workers = config.system.optimizer.num_workers)

    trainer.test(system, dataloaders = en_dl)
    en_results = system.test_results

    trainer.test(system, dataloaders = es_dl)
    es_results = system.test_results

    acc_diff = None
    # =============================
    # FILL ME OUT
    # 
    # Compute the difference in accuracy between two groups: 
    # english and spanish reviews. 
    # 
    # Pseudocode:
    # --
    # acc_diff = |english accuracy - spanish accuracy|
    # 
    # Type:
    # --
    # acc_diff: float (> 0 and < 1)
    # =============================

    print(f'[lambd={lambd}] Results on English reviews:')
    pprint(en_results)

    print(f'[lambd={lambd}] Results on Spanish reviews:')
    pprint(es_results)

    self.lambd = lambd
    self.acc_diff = acc_diff
    self.en_results = en_results
    self.es_results = es_results

    self.next(self.join)

  @step
  def join(self, inputs):
    index = None
    # =============================
    # FILL ME OUT
    # 
    # Calculate the index with the lowest difference in accuracy. 
    # 
    # Pseudocode:
    # --
    # Loop through inputs. Each input has a `acc_diff` param.
    # 
    # Type:
    # --
    # index: integer
    # 
    # Notes:
    # -- 
    # Our solution is 2 lines of code.
    # =============================

    en_results = inputs[index].en_results
    es_results = inputs[index].es_results
    best_lambd = inputs[index].lambd

    print(f'[best lambd={best_lambd}] Results on English reviews:')
    pprint(en_results)

    print(f'[best lambd={best_lambd}] Results on Spanish reviews:')
    pprint(es_results)

    log_file = join(LOG_DIR, 'jtt_flow', 'en_results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(en_results, log_file)  # save to disk

    log_file = join(LOG_DIR, 'jtt_flow', 'es_results.json')
    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(es_results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python jtt_flow.py`. To list
  this flow, run `python jtt_flow.py show`. To execute
  this flow, run `python jtt_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python jtt_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python jtt_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = JustTrainTwice()

