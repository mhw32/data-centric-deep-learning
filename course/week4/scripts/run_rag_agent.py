from os.path import join
from typing import List
from sentence_transformers import SentenceTransformer

from rag.utils import pc, pr, pb, load_config
from rag.llm import query_openai, get_welcome_message
from rag.prompts import get_persona, get_retrieval_prompt
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

    response: str = ""        # use this variable for the LLM response
    docs: List[str] = []      # use this variable for the document texts
    # ===========================
    # FILL ME OUT
    # 
    # Generate a LLM response while providing context from RAG.
    # Use the imported functions above to assist you
    # 
    # Steps:
    # 1. Compute embedding of query
    #    HINT: Take into account embedding model choice
    #    HINT: Convert embedding to a Python list 
    # 2. Retrieve most similar documents from Starpoint 
    #    HINT: Take into account text search weight and top k
    #    HINT: You will need to extract the document text from the returned Starpoint object
    #    Call this variable `contexts` which we will print below
    # 3. Build prompt 
    # 4. Call OpenAI to generate response
    embedding = embedding_model.encode(query).tolist()
    results = retrieve_documents(
      args.starpoint_api_key,
      collection_name=collection_name,
      query=query,
      query_embedding=embedding,
      top_k=config.top_K,
      text_search_weight=config.text_search_weight,
    )
    docs = [result['metadata']['description'] for result in results]
    prompt = get_retrieval_prompt(query, docs)
    response = query_openai(
      args.openai_api_key, 
      user_prompt=prompt, 
      system_prompt=system_prompt,
    )
    # ===========================
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