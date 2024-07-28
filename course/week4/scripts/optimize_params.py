import time
import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from os.path import join
from os import environ as env
from os import makedirs
from typing import List
from dotmap import DotMap
from dotenv import load_dotenv
from metaflow import FlowSpec, step, Parameter
from sentence_transformers import SentenceTransformer

from rag.paths import DATA_DIR
from rag.vector import retrieve_documents, get_my_collection_name

load_dotenv()


class OptimizeRagParams(FlowSpec):
  r"""MetaFlow to optimize a RAG hyperparameters by maximizing a retrieval 
  metric on top of an evaluation set.

  Arguments
  ---------
  questions_file (str, default: data/questions/questions.csv): path to generated questions CSV
  starpoint_api_key (str, default: env['STARPOINT_API_KEY']): Starpoint API key
  """
  questions_file = Parameter(
    'questions_file', 
    help = 'Path to generated questions', 
    default = join(DATA_DIR, 'questions/questions.csv'),
  )
  starpoint_api_key = Parameter(
    'starpoint_api_key', 
    help = 'Starpoint API key', 
    default = env['STARPOINT_API_KEY'],
  )

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
    hparams: List[DotMap] = []
    # ===========================
    # FILL ME OUT
    # Define a set of hyperparameters to search over. We want to compare
    # - two different embedding models: all-MiniLM-L6-v2 vs thenlper/gte-small
    # - two different text search weights: 0 vs 0.5
    # - whether to use hyde embeddings or question embeddings
    # In total this is searching over 8 configurations. In practice, we may search
    # over 100,000s but this should illustruate the point.
    # TODO
    # ===========================
    assert len(hparams) > 0, "Remember to complete the code in `get_search_space`"
    assert len(hparams) == 8, "You should have 8 configurations" 
    self.hparams = hparams
    self.next(self.optimize, foreach='hparams')

  @step 
  def optimize(self):
    r"""Compute retrieval accuracy.
    :param hparam: Hyperparameter for this RAG retrieval system
    """
    print(f'Evaluating configuration: {self.input}')
    # Load the questions CSV containing generated questions and the 
    # doc id used to generate that question.
    questions = pd.read_csv(self.questions_file)

    # Use this to retrieve documents
    collection_name = get_my_collection_name(
      env['GITHUB_USERNAME'],
      embedding=self.input.embedding, 
      hyde=self.input.hyde_embeddings,
    )
    embedding_model = SentenceTransformer(self.input.embedding)

    hits = 0
    for i in tqdm(range(len(questions))):
      question = questions.question.iloc[i]
      gt_id = questions.doc_id.iloc[i]
      # ===========================
      # FILL ME OUT
      # Write code to do the following:
      #   1. Perform top-3 retrieval using the question using Starpoint. To do this you will 
      #      need to use SentenceTransformers to compute embedding. Make sure to take into account
      #      the three hyperparameters in `hparams` to do this.
      #   2. Track if the correct document appears in the top 3 retrieved documents.
      #      +1 to `hits` if it does. +0 to `hits` if not.
      # TODO
      # ===========================

    hit_rate = hits / float(len(questions))
    self.hit_rate = hit_rate  # save to class
    self.hparam = self.input
    self.next(self.find_best)

  @step
  def find_best(self, inputs):
    r"""Given the outputs from the optimization, find the best hyperparameters
    by hit rate.
    """    
    save_dir = join(DATA_DIR, 'runs')
    makedirs(save_dir, exist_ok=True)

    results = []
    for input in inputs:
      result = {
        'hit_rate': input.hit_rate,
        **input.hparam,
      }
      results.append(result)

    results = pd.DataFrame.from_records(results)
    results.to_csv(join(save_dir, f'run-{int(time.time())}.csv'), index=False)
    
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
