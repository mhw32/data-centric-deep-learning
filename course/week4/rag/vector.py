from typing import List, Dict
from starpoint.db import Client


def get_my_collection_name(github_username: str, embedding: str = 'all-MiniLM-L6-v2', hyde: bool = False) -> str:
  r"""Helper function to return collection name.
  :note: All learners will be sharing an API key so it is important everyone has different collection names.
  """
  appendix = "-hyde" if hyde else ""
  embedding = embedding.replace('/', '-')
  return f"dcdl-week4-{github_username}-{embedding.lower()}{appendix}"


def create_collection(api_key: str, collection_name: str, dimensionality: int = 768) -> Dict:
  r"""Create a collection in Starpoint.
  :param api_key: Starpoint API key
  :param collection_name: Desired name of the collection. Please user `get_my_collection_name`
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


def delete_collection(api_key: str, collection_id: str) -> bool:
  r"""Delete a collection in Starpoint.
  :param api_key: Starpoint API key
  :param collection_id: ID for the collection (given to you on creation)
  :return: success
  """
  client = Client(api_key=api_key)
  result = client.delete_collection(collection_id=collection_id)
  return result.success


def insert_documents(
  api_key: str,
  collection_name: str,
  documents: List[Dict],
) -> Dict:
  r"""Insert documents into the documents.
  :param api_key: Starpoint API key
  :param collection_name: Desired name of the collection. Please set this to `dcdl-week4-<YOUR_GITHUB_USERNAME>`.
  :return: See https://docs.starpoint.ai/create-documents.
  {
    collection_id: string,
    documents: { id: string }
  }
  """
  starpoint = Client(api_key=api_key)
  result = starpoint.insert(documents=documents, collection_name=collection_name)
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
