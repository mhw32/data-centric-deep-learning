import torch
import random
import numpy as np
from os.path import join
from metaflow import FlowSpec, step, Parameter

from rag.paths import DATA_DIR, LOG_DIR, CONFIG_DIR


class OptimizeRAGParams(FlowSpec):
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

    self.next(self.end)

  @step
  def end(self):
    r"""End node!"""
    print('done! great work!')


if __name__ == "__main__":
  """
  To validate this flow, run `python optimize_params.py`. To list
  this flow, run `python optimize_params.py show`. To execute
  this flow, run `python optimize_params.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python optimize_params.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python optimize_params.py resume`
  
  You can specify a run id as well.
  """
  flow = OptimizeRAGParams()
