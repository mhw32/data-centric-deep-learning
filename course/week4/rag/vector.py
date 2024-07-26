from typing import List, Dict
from starpoint.db import Client
from sentence_transformers import SentenceTransformer


def retrieve_documents(
  api_key: str,
  collection_id: str,
  user_query: str,
  embedding_model: SentenceTransformer,
  top_k: int = 1,
  embedding_dim: int = 768,
  text_search_weight: float = 0.5,
) -> List[Dict]:
  r"""Query starpoint to get the top K documents.
  :return docs: List of k documents relevant to the query
  """
  starpoint = Client(api_key=api_key)
  embedding = embedding_model.encode(user_query)
  embedding: List[float] = embedding.tolist()
  response = starpoint.query(
    collection_id=collection_id,
    query_embedding={
      "values": embedding,
      "dimensionality": embedding_dim,
    },
    top_k=top_k,
    text_search_query=[user_query],
    text_search_weight=text_search_weight,
  )
  results = response["results"]
  return results