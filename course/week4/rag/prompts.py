from typing import List


def get_retrieval_prompt(query: str, documents: List[str]) -> str:
  context = '\n\n'.join(documents)
  prompt = f'''
Answer the question using only the provided context.
Refrain from including information that is not provided in the context.
Your answer should be in your own words and be no longer than 50 words. 

CONTEXT: {context}

INPUT: {query}

ANSWER:                             
'''
  return prompt


def get_question_prompt(context: str):
  prompt = f"""Generate a single question that can be answered solely from the provided context.

CONTEXT: {context}

QUESTION:
"""
  return prompt


def get_hyde_response_prompt(question: str) -> str:
  prompt = f"""Please answer the following question to the best of your ability. 
If you do not know the answer, make your best guess. Do not say you do not know the answer. 

QUESTION: {question}

ANSWER: 
"""
  return prompt


def get_question_judge_prompt(question: str, context: str) -> str:
  # https://lightning.ai/panchamsnotes/studios/evaluate-your-rag-part-1-synthesize-an-evaluation-dataset
  prompt = f"""You will be given a context and a question.
Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

CONTEXT: {context}

QUESTION: {question}

RATING: 
"""
  return prompt


def get_response_judge_prompt(question: str, response: str) -> str:
  # Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
  # https://arxiv.org/abs/2306.05685
  prompt = f"""Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. 
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. 
Give your answer on a scale of 1 to 5, where 1 means that the repsonse does not answer question at all, and 5 means that the response clearly and unambiguously answers the question.

QUESTION: {question}

RESPONSE: {response}

EXPLANATION: 

RATING:
"""  
  return prompt
