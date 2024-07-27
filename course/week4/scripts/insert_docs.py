from tqdm import tqdm
from os.path import join, isfile
from sentence_transformers import SentenceTransformer

from rag.llm import embedding_name_to_dim
from rag.vector import get_my_collection_name, insert_documents
from rag.utils import load_config
from rag.dataset import load_documents
from rag.paths import CONFIG_DIR


def main(args):
  r"""Inserts new documents in the collection.
  """
  assert isfile(args.documents), f'Documents file {args.documents} does not exist'
  config = load_config(args.config)
  collection_name = get_my_collection_name(
    config.github_username, 
    embedding=config.embedding, 
    hyde=config.hyde_embeddings,
  )

  # Load raw documents as a Pandas Dataframe with two columns
  # - doc_id: Document ID 
  # - text: Content for the document
  raw = load_documents()

  # Use the embedding model to embed docs
  embedding_dim = embedding_name_to_dim(config.embedding)
  embedding_model = SentenceTransformer(config.embedding)

  documents = []
  # ===========================
  # FILL ME OUT
  # Prepare the documents to be inserted into the vector db
  # You will need compute embeddings. Make sure to cast the embedding to a list.
  # Please refer to `config.json` for which embedding to use:
  # Example document:
  # {
  #   "embeddings": {
  #     "values": [0.1, 0.2, 0.3, 0.4, 0.5],
  #     "dimensionality": 5,
  #   }, # single vector document
  #   "metadata": {
  #     "doc_id": "...",
  #   }
  # }
  # Please add the document ID to the metadata under the key `doc_id`.
  # Please see docs here: https://docs.starpoint.ai/create-documents 
  for i in tqdm(range(len(raw))):
    text = raw.text[i]
    doc_id = raw.doc_id[i]
    embedding = embedding_model.encode(text).tolist()
    doc = {
      "embeddings": {
        "values": embedding,
        "dimensionality": embedding_dim,
      },
      "metadata": {
        "doc_id": doc_id,
      }
    }
    documents.append(doc)
  # ===========================

  assert len(documents) > 0, f"Please remember to append to the documents array"
  print(f'inserting documents into Starpoint collection {collection_name}')
  insert_documents(args.starpoint_api_key, collection_name, documents)
  print(f'Done. {len(documents)} inserted.')


if __name__ == "__main__":
  from os import environ as env
  from dotenv import load_dotenv
  load_dotenv()

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('docs_csv', type=str, help='Path to documents to insert')
  parser.add_argument('--config', type=str, default=join(CONFIG_DIR, 'insert.json'), help='Configuration file')
  parser.add_argument('--starpoint-api-key', type=str, default=env['STARPOINT_API_KEY'], help='Starpoint API key')
  args = parser.parse_args()

  main(args)