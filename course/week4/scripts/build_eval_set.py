import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from os import makedirs
from os import environ as env
from os.path import join, isfile, splitext
from dotenv import load_dotenv
from metaflow import FlowSpec, step, Parameter

from rag.llm import query_openai
from rag.prompts import get_question_prompt, get_question_judge_prompt, get_hyde_response_prompt
from rag.dataset import chunk_document, load_documents
from rag.paths import DATA_DIR

load_dotenv()


class BuildEvaluationSet(FlowSpec):
  r"""MetaFlow to build an evaluation set of questions used to compute retrieval 
  scores for a RAG system.

  Arguments
  ---------
  config (str, default: configs/eval.json): path to a configuration file
  """
  openai_api_key = Parameter('openai_api_key', help = 'OpenAI API key', default=env['OPENAI_API_KEY'])

  @step
  def start(self):
    r"""Start node.
    Set random seeds for reproducibility.
    """
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    self.next(self.write_questions)

  @step
  def write_questions(self):
    r"""Write questions to use as an evaluation set.
    """
    docs = load_documents()
    num_docs = len(docs)

    # Loop through questions 
    questions: List[str] = []
    contexts: List[str] = []
    doc_ids: List[str] = [] # saves the original doc_id
    for i in tqdm(num_docs, desc="Writing questions"):
      text = str(docs.iloc[i].text)
      doc_id = str(docs.iloc[i].doc_id)
      # Convert the text into chunks so we can write a question about each chunk
      # This funciton groups each 
      chunks = chunk_document(text)
      for chunk in chunks:
        question = ""
        # ===========================
        # FILL ME OUT
        # Use `query_openai` to generate a question from the chunk text `chunk`. 
        # See `rag/prompts` for a bank of relevant prompts to use. You may edit any prompts in there.
        # Save the generated question (as a string) into the `question` variable.
        question = query_openai(self.openai_api_key, get_question_prompt(chunk))
        # ===========================
        assert len(question) > 0, f"Did you complete the coding section in `write_questions`?"
        questions.append(question)
        doc_ids.append(doc_id) # save the doc id for each 
        contexts.append(chunk) # save chunk used to gen question

    assert len(questions) == len(contexts), f"Mismatch in size. Got {len(questions)} != {len(contexts)}."
    print(f'Wrote {len(questions)} questions.')

    self.contexts = contexts
    self.questions = questions
    self.doc_ids = doc_ids
    self.next(self.grade_questions)

  @step
  def grade_questions(self):
    r"""Use an LLM judge to grade each question and toss anything below a rating of 3.
    """
    ratings: List[int] = []
    for i in tqdm(range(self.questions), desc="Grading questions"):
      rating = -1
      # ===========================
      # FILL ME OUT
      # Use `query_openai` to ask an LLM judge to grade if a generated question fits the context.
      # See `rag/prompts` for a bank of relevant prompts to use. You may edit any prompts in there.
      # Make sure the response is an integer. 
      # If the response is not castable as an integer, skip the question.
      rating = query_openai(self.openai_api_key, get_question_judge_prompt(self.questions[i], self.contexts[i]))
      rating = int(rating)
      # ===========================
      assert rating > 0, f"Did you complete the coding section in `grade_questions`?"
      ratings.append(rating)

    assert len(self.questions) == len(ratings), \
      f"Mismatch in size. Got {len(self.questions)} questions != {len(ratings)} ratings."
    print(f'Average rating: {np.mean(ratings)}')

    self.ratings = ratings
    self.next(self.write_hypothetical_answers)

  @step
  def write_hypothetical_answers(self):
    r"""We will want to explore hyde embeddings. To do that, we need to generate short
    hypothetical answers to these questions.
    """
    hypo_answers: List[str] = []
    for i in tqdm(range(self.questions), desc="Writing questions"):
      hypo_answer = ""
      # ===========================
      # FILL ME OUT
      # Use `query_openai` to write a short answer to each question.
      # See `rag/prompts` for a bank of relevant prompts to use. You may edit any prompts in there.
      hypo_answer = query_openai(self.openai_api_key, get_hyde_response_prompt(self.questions[i]))
      # ===========================
      assert len(hypo_answer) > 0, f"Did you complete the coding section in `write_hypothetical_answers`?"
      hypo_answers.append(hypo_answer)

    assert len(self.questions) == len(hypo_answers), \
      f"Mismatch in size. Got {len(self.questions)} questions != {len(hypo_answers)} hypothetical answers."
    print(f'Wrote {len(hypo_answers)} hypothetical answers.')

    self.hypo_answers = hypo_answers
    self.next(self.save_questions)

  @step
  def save_questions(self):
    r"""Save questions to `data/questions`.
    """
    dataset_name, _ = splitext(self.config.dataset)
    makedirs(join(DATA_DIR, 'questions'), exist_ok=True)
    question_file = join(DATA_DIR, f'questions/{dataset_name}.csv')
    # Make a dataframe and save it
    dataset = {
      'doc_id': self.doc_ids, 
      'context': self.contexts, 
      'question': self.questions, 
      'ratings': self.ratings,
      'hypo_answers': self.hypo_answers,
    }
    dataset = pd.DataFrame(dataset)
    dataset.to_csv(question_file, index=False)

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
