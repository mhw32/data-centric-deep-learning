from typing import List


def get_retrieval_prompt(user_query: str, documents: List[str]) -> str:
  context = '\n\n'.join(documents)
  prompt = f'''
Answer the question using only the provided context.
Refrain from including information that is not provided in the context.
Your answer should be in your own words and be no longer than 50 words. 

CONTEXT: {context}

INPUT: {user_query}

ANSWER:                             
'''
  return prompt


def get_question_prompt(context: str):
  prompt = f"""Generate a single question that can be answered solely from the provided context.

CONTEXT: {context}

QUESTION:
"""
  return prompt


def craft_hyde_prompt(question: str) -> str:
  prompt = f"""Please answer the following question to the best of your ability. 
If you do not know the answer, make your best guess. Do not say you do not know the answer. 

QUESTION: {question}

ANSWER: 
"""
  return prompt


def get_llm_judge_prompt(question: str, response: str) -> str:
  # Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena
  # https://arxiv.org/abs/2306.05685
  prompt = f"""Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. 
Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. 
Begin your evaluation by providing a short explanation. Be as objective as possible. 
After providing your explanation, please rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

QUESTION: {question}

RESPONSE: {response}

EXPLANATION: 

RATING:
"""  
  return prompt


def get_persona() -> str:
  return ""  # TODO
