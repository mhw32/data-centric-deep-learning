import torch
import random
import numpy as np
from os.path import join
from typing import Dict, List
from dotmap import DotMap
from metaflow import FlowSpec, step, Parameter

from rag.paths import DATA_DIR, LOG_DIR, CONFIG_DIR


class OptimizeRagParams(FlowSpec):
  r"""MetaFlow to optimize a RAG hyperparameters by maximizing a retrieval 
  metric on top of an evaluation set.

  Arguments
  ---------
  config (str, default: configs/optim.json): path to a configuration file
  """
  config_path = Parameter('config', help = 'path to config file', default=join(CONFIG_DIR, 'optim.json'))

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.get_search_space)

  @step 
  def get_search_space(self):
    r"""Define a set of RAG configurations to search over.
    """
    # Each hyperparameter setting should loo like:
    #   hparam = DotMap({
    #     "embedding": "all-MiniLM-L6-v2",
    #     "text_search_weight": 0.5,
    #     "hyde_embeddings": False,
    #   })
    self.hparams: List[DotMap] = []
    # ===========================
    # FILL ME OUT
    # Define a set of hyperparameters to search over. We want to compare
    # - two different embedding models: all-MiniLM-L6-v2 vs thenlper/gte-small
    # - two different text search weights: 0 vs 0.5
    # - whether to use hyde embeddings or question embeddings
    # In total this is searching over 8 configurations. In practice, we may search
    # over 100,000s but this should illustruate the point.
    # ===========================
    self.next(self.optimize, foreach='hparams')

  @step 
  def optimize(self, hparam: DotMap):
    r"""Compute retrieval accuracy.
    :param hparam: Hyperparameter for this RAG retrieval system
    """
    self.next(self.find_best)

  @step
  def find_best(self, inputs):
    r"""Given the outputs from the optimization, find the best hyperparameters
    by hit rate.
    """
    self.next(self.end)

  @step
  def end(self):
    r"""End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python optimize_params.py`. To list
  this flow, run `python optimize_params.py show`. To execute
  this flow, run `python optimize_params.py --max-workers 1 run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python optimize_params.py --no-pylint --max-workers 1 run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python optimize_params.py resume`
  
  You can specify a run id as well.
  """
  flow = OptimizeRagParams()
