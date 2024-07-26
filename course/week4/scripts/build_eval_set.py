import torch
import random
import numpy as np
from os.path import join
from metaflow import FlowSpec, step, Parameter

from rag.paths import DATA_DIR, LOG_DIR, CONFIG_DIR


class BuildEvaluationSet(FlowSpec):
  r"""MetaFlow to build an evaluation set of questions used to compute retrieval 
  scores for a RAG system.

  Arguments
  ---------
  config (str, default: configs/eval.json): path to a configuration file
  """
  config_path = Parameter('config', help = 'path to config file', default=join(CONFIG_DIR, 'eval.json'))

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
  To validate this flow, run `python build_eval_set.py`. To list
  this flow, run `python build_eval_set.py show`. To execute
  this flow, run `python build_eval_set.py run`.

  You may get PyLint errors from `numpy.random`. If so,
  try adding the flag:

    `python build_eval_set.py --no-pylint run`

  If you face a bug and the flow fails, you can continue
  the flow at the point of failure:

    `python build_eval_set.py resume`
  
  You can specify a run id as well.
  """
  flow = BuildEvaluationSet()
