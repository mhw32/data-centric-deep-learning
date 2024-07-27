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

from rag.paths import DATA_DIR, CONFIG_DIR
from rag.vector import retrieve_documents, get_my_collection_name

load_dotenv()


class OptimizeRagParams(FlowSpec):
  r"""MetaFlow to optimize a RAG hyperparameters by maximizing a retrieval 
  metric on top of an evaluation set.

  Arguments
  ---------
  config (str, default: configs/optim.json): path to a configuration file
  """
  config_path = Parameter('config', help = 'path to config file', default=join(CONFIG_DIR, 'optim.json'))
  starpoint_api_key = Parameter('starpoint_api_key', help = 'Starpoint API key', default=env['STARPOINT_API_KEY'])

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
    for embedding in ['all-MiniLM-L6-v2', 'thenlper/gte-small']:
      for text_search_weight in [0, 0.5]:
        for hyde_embeddings in [True, False]:
          hparam = DotMap({
            "embedding": embedding,
            "text_search_weight": text_search_weight,
            "hyde_embeddings": hyde_embeddings,
          })
          hparams.append(hparam)
    # ===========================
    assert len(hparams) > 0, f"Remember to complete the code in `get_search_space`"
    self.hparams = hparams
    self.next(self.optimize, foreach='hparams')

  @step 
  def optimize(self, hparam: DotMap):
    r"""Compute retrieval accuracy.
    :param hparam: Hyperparameter for this RAG retrieval system
    """
    # Load the questions CSV containing generated questions and the 
    # doc id used to generate that question.
    questions = pd.read_csv(join(DATA_DIR, 'questions', self.config.questions))

    # Use this to retrieve documents
    collection_name = get_my_collection_name(self.config.github_username)
    embedding_model = SentenceTransformer(hparam.embedding)

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
      embedding = embedding_model.encode(question).tolist()
      results = retrieve_documents(
        self.starpoint_api_key, 
        collection_name, 
        question, 
        embedding,
        top_k=3, 
        text_search_weight=hparam.text_search_weight,
      )
      retrieved_ids = [result['metadata']['metadata']['doc_id'] for result in results]
      if gt_id in retrieved_ids:
        hits += 1
      # ===========================

    hit_rate = hits / float(len(questions))
    self.hit_rate = hit_rate  # save to class
    self.hparam = hparam
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
