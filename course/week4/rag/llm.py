import openai
from typing import Optional


def query_openai(
  api_key: str,
  user_prompt: str,
  system_prompt: Optional[str] = None,
  model: str = "gpt-3.5-turbo",
  api_type: str = "openai",
  response_format: str = "text",
) -> str:
  r"""Query OpenAI to generate a response to a query.
  :param api_key (str): API key for OpenAI
  """
  messages = [{"role": "user", "content": user_prompt}]
  if system_prompt is not None:
    messages.append({"role": "system", "content": system_prompt})
  openai.api_key = api_key
  openai.api_type = api_type
  completion = openai.chat.completions.create(
      model=model,
      response_format={"type": response_format},
      messages=messages,
  )
  answer = completion.choices[0].message.content
  return answer


def get_welcome_message() -> str:
  r"""This is the default message to send when the RAG-LLM agent boots up.
  """
  message = "Hi there! I'm an AI assistant to help answer your questions. How can I help?"
  return message
