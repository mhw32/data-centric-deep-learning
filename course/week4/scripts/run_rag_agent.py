from sentence_transformers import SentenceTransformer

from rag.utils import pc, pr, pb
from rag.llm import query_openai, get_welcome_message
from rag.prompts import get_persona, get_retrieval_prompt, get_hyde_response_prompt
from rag.vector import get_my_collection_name, retrieve_documents


def main(args):
  r"""Initialize a RAG agent and have a conversation with it. 
  """
  system_prompt = get_persona()
  collection_name = get_my_collection_name(env['GITHUB_USERNAME'])
  embedding_model = SentenceTransformer(args.embedding)

  pc(get_welcome_message())
  while True:
    query = input("Type a message: ")
    pb(f'USER: {query}')

    if args.hyde:
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
      top_k=3,
      text_search_weight=args.text_search_weight,
    )
    docs = [result['metadata']['text'] for result in results]
    response = query_openai(
      args.openai_api_key, 
      user_prompt=get_retrieval_prompt(query, docs), 
      system_prompt=system_prompt,
    )

    for i, doc in enumerate(docs):
      pr(f'DOC #{i}: {doc[:100]}...')
    pc(f'ASSISTANT: {response}')


if __name__ == "__main__":
  from os import environ as env
  from dotenv import load_dotenv
  load_dotenv()

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--embedding', type=str, default='all-MiniLM-L6-v2', help='Embedding to use (default: all-MiniLM-L6-v2)')
  parser.add_argument('--hyde', action='store_true', default=False, help='Use hyde embeddings (default: False)')
  parser.add_argument('--text-search-weight', type=float, default=0.5, help='Weight between lexical and semantic search (default: 0.5)')
  parser.add_argument('--openai-api-key', type=str, default=env['OPENAI_API_KEY'], help='OpenAI API key')
  parser.add_argument('--starpoint-api-key', type=str, default=env['STARPOINT_API_KEY'], help='Starpoint API key')
  args = parser.parse_args()

  main(args)