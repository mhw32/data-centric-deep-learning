from os.path import join
from typing import List
from sentence_transformers import SentenceTransformer

from rag.utils import pc, pr, pb, load_config
from rag.llm import query_openai, get_welcome_message
from rag.prompts import get_persona, get_retrieval_prompt, get_hyde_response_prompt
from rag.vector import get_my_collection_name, retrieve_documents
from rag.paths import CONFIG_DIR


def main(args):
  r"""Initialize a RAG agent and have a conversation with it. 
  """
  collection_name = get_my_collection_name(args.github_username)
  system_prompt = get_persona()

  config = load_config(args.config)
  embedding_model = SentenceTransformer(config.embedding)

  pc(get_welcome_message())
  while True:
    query = input("Type a message: ")
    pb(f'USER: {query}')

    if config.hyde_embeddings:
      # If we use hyde embeddings, then we need to embed the hypothetical 
      # answer instead of the question
      hypo_answer = query_openai(args.openai_api_key, get_hyde_response_prompt(query))
      embedding = embedding_model.encode(hypo_answer).tolist()  
    else:
      embedding = embedding_model.encode(query).tolist()  
    
    results = retrieve_documents(
      args.starpoint_api_key,
      collection_name=collection_name,
      query=query,  # regardless of hyde, keep query here
      query_embedding=embedding,
      top_k=config.top_k,
      text_search_weight=config.text_search_weight,
    )
    docs = [result['metadata']['description'] for result in results]
    response = query_openai(
      args.openai_api_key, 
      user_prompt=get_retrieval_prompt(query, docs), 
      system_prompt=system_prompt,
    )

    for i, doc in enumerate(docs):
      pr(f'DOC #{i}: {doc}')
    pc(f'ASSISTANT: {response}')


if __name__ == "__main__":
  from os import environ as env
  from dotenv import load_dotenv
  load_dotenv()

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, default=join(CONFIG_DIR, 'rag.json'), help='Configuration file')
  parser.add_argument('--openai-api-key', type=str, default=env['OPENAI_API_KEY'], help='OpenAI API key')
  parser.add_argument('--starpoint-api-key', type=str, default=env['STARPOINT_API_KEY'], help='Starpoint API key')
  args = parser.parse_args()

  main(args)