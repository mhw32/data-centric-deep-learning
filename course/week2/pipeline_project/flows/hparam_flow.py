"""
Flow #2: This flow will train a multilayer perceptron with a 
width found by hyperparameter search.
"""

import os
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

from src.system import MNISTDataModule, DigitClassifierSystem
from src.utils import load_config, to_json


class DigitClassifierFlow(FlowSpec):
  r"""A flow that trains a image classifier to recognize handwritten
  digit, such as those in the MNIST dataset. We search over three 
  potential different widths.

  Arguments
  ---------
  config (str, default: ./config.py): path to a configuration file
  """ 
  config_path = Parameter('config', 
    help = 'path to config file', default='./configs/hparam_flow.json')

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.widths = [16, 32, 64]

    # for each width, we will train a model
    self.next(self.init_and_train, foreach='widths')

  @step
  def init_and_train(self):
    r"""Instantiates a data module, pytorch lightning module, 
    and lightning trainer instance. Calls `fit` on the trainer.

    This should remind of you of `init_system` and `train_model`
    from Flow #1. We merge the two into one node.
    """
    config = load_config(self.config_path)
    config.system.model.width = self.input

    dm = MNISTDataModule(config)
    system = DigitClassifierSystem(config)

    checkpoint_callback = ModelCheckpoint(
      dirpath = os.path.join(config.system.save_dir, f'width{self.input}'),
      monitor = 'dev_loss',
      mode = 'min',
      save_top_k = 1,
      verbose = True,
    )

    trainer = Trainer(
      max_epochs = config.system.optimizer.max_epochs,
      callbacks = [checkpoint_callback])

    trainer.fit(system, dm)

    self.dm = dm
    self.system = system
    self.trainer = trainer
    self.callback = checkpoint_callback

    self.next(self.find_best)

  @step
  def find_best(self, inputs):
    r"""Only keep the system with the lowest `dev_loss."""

    # manually propagate class variables through but we only
    # need a few of them so no need to call `merge_artifacts`
    self.dm = inputs[0].dm

    scores = []        # populate with scores from each hparams
    best_index = None  # replace with best index
    
    # ================================
    # FILL ME OUT
    # 
    # Aggregate the best validation performance across inputs into
    # the variable `scores`.
    # 
    # HINT: the `callback` object has a property `best_model_score`
    #       that make come in handy. 
    # 
    # Then, compute the index of the model and store it in `best_index`.
    # 
    # Pseudocode:
    # --
    # aggregate scores using `inputs`
    # best_index = ...
    #
    # Type:
    # --
    # scores: List[float] 
    # best_index: integer 
    # ================================

    # sanity check for scores length
    assert len(scores) == len(list(inputs)), "Hmm. Incorrect length for scores."
    # sanity check for best_index
    assert best_index is not None
    assert best_index >= 0 and best_index < len(list(inputs))
    
    # get the best system / trainer
    # we drop the callback
    self.system = inputs[best_index].system
    self.trainer = inputs[best_index].trainer

    # save the best width
    self.best_width = inputs[best_index].widths[best_index]

    # only the best model proceeds to offline test
    self.next(self.offline_test)

  @step
  def offline_test(self):
    r"""Calls (offline) `test` on the trainer. Saves results to a log file."""

    self.trainer.test(self.system, self.dm, ckpt_path = 'best')
    results = self.system.test_results
    results['best_width'] = self.best_width

    pprint(results)

    log_file = join(Path(__file__).resolve().parent.parent, 
      f'logs/hparam_flow', 'offline-test-results.json')

    os.makedirs(os.path.dirname(log_file), exist_ok = True)
    to_json(results, log_file)  # save to disk

    self.next(self.end)  

  @step
  def end(self):
    """End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python hparam_flow.py`. To list
  this flow, run `python hparam_flow.py show`. To execute
  this flow, run `python hparam_flow.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python hparam_flow.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python hparam_flow.py resume`
  
  You can specify a run id as well.
  """
  flow = DigitClassifierFlow()
