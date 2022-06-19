"""This flow will train a neural network to perform sentiment classification 
for the beauty products reviews.
"""

import os
import torch
import random
import numpy as np
import pandas as pd
from os.path import join
from pathlib import Path
from pprint import pprint
from torch.utils.data import DataLoader, TensorDataset

from metaflow import FlowSpec, step, Parameter
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from cleanlab.filter import find_label_issues
from sklearn.model_selection import KFold

from src.system import ReviewDataModule, SentimentClassifierSystem
from src.utils import load_config, to_json
from src.consts import DATA_DIR


class TrainIdentifyReview(FlowSpec):
  r"""A MetaFlow that trains a sentiment classifier on reviews of luxury beauty
  products using PyTorch Lightning, identifies data quality issues using CleanLab, 
  and prepares them for review in LabelStudio.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """
  config_path = Parameter('config', 
    help = 'path to config file', default='./config.json')

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
      dirpath = config.train.ckpt_dir,
      monitor = 'dev_loss',
      mode = 'min',    # look for lowest `dev_loss`
      save_top_k = 1,  # save top 1 checkpoints
      verbose = True,
    )

    trainer = Trainer(
      max_epochs = config.train.optimizer.max_epochs,
      callbacks = [checkpoint_callback])

    # when we save these objects to a `step`, they will be available
    # for use in the next step, through not steps after.
    self.dm = dm
    self.system = system
    self.trainer = trainer
    self.config = config

    self.next(self.train_test)

  @step
  def train_test(self):
    """Calls `fit` on the trainer.
    
    We first train and (offline) evaluate the model to see what 
    performance would be without any improvements to data quality.
    """
    # Call `fit` on the trainer with `system` and `dm`.
    # Our solution is one line.
    self.trainer.fit(self.system, self.dm)
    self.trainer.test(self.system, self.dm, ckpt_path = 'best')

    # results are saved into the system
    results = self.system.test_results

    # print results to command line
    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs', 'pre-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.crossval)
  
  @step
  def crossval(self):
    r"""Confidence learning requires cross validation to compute 
    out-of-sample probabilities for every element. Each element
    will appear in a single cross validation split exactly once. 
    """
    # combine training and dev datasets
    X = np.concatenate([
      np.asarray(self.dm.train_dataset.embedding),
      np.asarray(self.dm.dev_dataset.embedding),
      np.asarray(self.dm.test_dataset.embedding),
    ])
    y = np.concatenate([
      np.asarray(self.dm.train_dataset.data.label),
      np.asarray(self.dm.dev_dataset.data.label),
      np.asarray(self.dm.test_dataset.data.label),
    ])

    probs = np.zeros(len(X))  # we will fill this in

    # https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    kf = KFold(n_splits=3)    # create kfold splits

    for train_index, test_index in kf.split(X):
      probs_ = None
      # ===============================================
      # FILL ME OUT
      # 
      # Fit a new `SentimentClassifierSystem` on the split of 
      # `X` and `y` defined by the current `train_index` and
      # `test_index`. Then, compute predicted probabilities on 
      # the test set. Store these probabilities as a 1-D numpy
      # array `probs_`.
      # 
      # Use `self.config.train.optimizer` to specify any hparams 
      # like `batch_size` or `epochs`.
      #  
      # HINT: `X` and `y` are currently numpy objects. You will 
      # need to convert them to torch tensors prior to training. 
      # You may find the `TensorDataset` class useful. Remember 
      # that `Trainer.fit` and `Trainer.predict` take `DataLoaders`
      # as an input argument.
      # 
      # Our solution is ~15 lines of code.
      # 
      # Pseudocode:
      # --
      # Get train and test slices of X and y.
      # Convert to torch tensors.
      # Create train/test datasets using tensors.
      # Create train/test data loaders from datasets.
      # Create `SentimentClassifierSystem`.
      # Create `Trainer` and call `fit`.
      # Call `predict` on `Trainer` and the test data loader.
      # Convert probabilities back to numpy (make sure 1D).
      # 
      # Types:
      # --
      # probs_: np.array[float] (shape: |test set|)
      # ===============================================
      assert probs_ is not None, "`probs_` is not defined."
      probs[test_index] = probs_

    # create a single dataframe with all input features
    all_df = pd.concat([
      self.dm.train_dataset.data,
      self.dm.dev_dataset.data,
      self.dm.test_dataset.data,
    ])
    all_df = all_df.reset_index(drop=True)
    # add out-of-sample probabilities to the dataframe
    all_df['prob'] = probs

    # save to excel file
    all_df.to_csv(join(DATA_DIR, 'prob.csv'), index=False)

    self.all_df = all_df
    self.next(self.inspect)

  @step
  def inspect(self):
    r"""Use confidence learning over examples to identify labels that 
    likely have issues with the `cleanlab` tool. 
    """
    prob = np.asarray(self.all_df.prob)
    prob = np.stack([1 - prob, prob]).T
  
    # rank label indices by issues
    ranked_label_issues = None
    
    # =============================
    # FILL ME OUT
    # 
    # Apply confidence learning to labels and out-of-sample
    # predicted probabilities. 
    # 
    # HINT: use cleanlab. See tutorial. 
    # 
    # Our solution is one function call.
    # 
    # Types
    # --
    # ranked_label_issues: List[int]
    # =============================
    assert ranked_label_issues is not None, "`ranked_label_issues` not defined."

    # save this to class
    self.issues = ranked_label_issues
    print(f'{len(ranked_label_issues)} label issues found.')

    # overwrite label for all the entries in all_df
    for index in self.issues:
      label = self.all_df.loc[index, 'label']
      # we FLIP the label!
      self.all_df.loc[index, 'label'] = 1 - label

    self.next(self.review)

  @step
  def review(self):
    r"""Format the data quality issues found such that they are ready to be 
    imported into LabelStudio. We expect the following format:

    [
      {
        "data": {
          "text": <review text>
        },
        "predictions": [
          {
            "value": {
              "choices": [
                  "Positive"
              ]
            },
            "from_name": "sentiment",
            "to_name": "text",
            "type": "choices"
          }
        ]
      }
    ]

    See https://labelstud.io/guide/predictions.html#Import-pre-annotations-for-text.and

    You do not need to complete anything in this function. However, look through the 
    code and make sure the operations and output make sense.
    """
    outputs = []
    for index in self.issues:
      row = self.all_df.iloc[index]
      output = {
        'data': {
          'text': str(row.review),
        },
        'predictions': [{
          'result': [
            {
              'value': {
                'choices': [
                  'Positive' if row.label == 1 else 'Negative'
                ]
              },
              'id': f'data-{index}',
              'from_name': 'sentiment',
              'to_name': 'text',
              'type': 'choices',
            },
          ],
        }],
      }
      outputs.append(output)

    # save to file
    preanno_path = join(self.config.review.save_dir, 'pre-annotations.json')
    to_json(outputs, preanno_path)

    self.next(self.retrain_retest)

  @step
  def retrain_retest(self):
    r"""Retrain without reviewing. Let's assume all the labels that 
    confidence learning suggested to flip are indeed erroneous."""
    dm = ReviewDataModule(self.config)
    train_size = len(dm.train_dataset)
    dev_size = len(dm.dev_dataset)

    # ====================================
    # FILL ME OUT
    # 
    # Overwrite the dataframe in each dataset with `all_df`. Make sure to 
    # select the right indices. Since `all_df` contains the corrected labels,
    # training on it will incorporate cleanlab's re-annotations.
    # 
    # Pseudocode:
    # --
    # dm.train_dataset.data = training slice of self.all_df
    # dm.dev_dataset.data = dev slice of self.all_df
    # dm.test_dataset.data = test slice of self.all_df
    # # ====================================

    # start from scratch
    system = SentimentClassifierSystem(self.config)
    trainer = Trainer(
      max_epochs = self.config.train.optimizer.max_epochs)

    trainer.fit(system, dm)
    trainer.test(system, dm, ckpt_path = 'best')
    results = system.test_results

    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs', 'post-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python flow_conflearn.py`. To list
  this flow, run `python flow_conflearn.py show`. To execute
  this flow, run `python flow_conflearn.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python flow_conflearn.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python flow_conflearn.py resume`
  
  You can specify a run id as well.
  """
  flow = TrainIdentifyReview()
