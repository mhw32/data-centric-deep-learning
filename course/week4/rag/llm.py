import openai


def query_openai(
  api_key: str,
  prompt: str,
  model: str = "gpt-4o",
  api_type: str = "openai",
  response_format: str = "text",
) -> str:
  r"""Query OpenAI to generate a response to a query.
  :param api_key (str): API key for OpenAI
  """
  openai.api_key = api_key
  openai.api_type = api_type
  completion = openai.chat.completions.create(
      model=model,
      response_format={"type": response_format},
      messages=[{"role": "user", "content": prompt}],
  )
  answer = completion.choices[0].message.content
  return answer



