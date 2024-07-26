from typing import List, Dict
from starpoint.db import Client


def create_collection(api_key: str, collection_name: str, dimensionality: int = 768) -> Dict:
  r"""Create a collection in Starpoint.
  :param api_key: Starpoint API key
  :param collection_name: Desired name of the collection. Please set this to `dcdl-week4-<YOUR_GITHUB_USERNAME>`.
  :param dimensionality: Expected number of embedding dimensions
  :note: See https://docs.starpoint.ai/create-documents/
  :return: 
  {
    id: string,
    name: string,
    dimensionality: int
  }
  """
  client = Client(api_key=api_key)
  result = client.create_collection(collection_name=collection_name, dimensionality=dimensionality)
  return result


def retrieve_documents(
  api_key: str,
  collection_name: str,
  query: str,
  query_embedding: List[float],
  top_k: int = 1,
  text_search_weight: float = 0.5,
) -> List[Dict]:
  r"""Query starpoint to get the top K documents.
  :param api_key: Starpoint API key
  :param collection_name: Desired name of the collection. Please set this to `dcdl-week4-<YOUR_GITHUB_USERNAME>`.
  :param query: String query 
  :param embedding: Embedded query. Make sure this is a python list, not a numpy array or torch tensor.
  :param top_k: Number of documents to fetch (default: 1)
  :param text_search_weight: Hyperparameter weight between lexical and semantic search
  :return docs: List of k documents relevant to the query
  """
  starpoint = Client(api_key=api_key)
  response = starpoint.query(
    collection_name=collection_name,
    query_embedding={
      "values": query_embedding,
      "dimensionality": len(query_embedding),
    },
    top_k=top_k,
    text_search_query=[query],
    text_search_weight=text_search_weight,
  )
  docs = response["results"]
  return docs
