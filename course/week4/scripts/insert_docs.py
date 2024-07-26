import pandas as pd
from os.path import join, isfile
from rag.vector import get_my_collection_name, insert_documents
from rag.utils import to_json, load_config
from rag.paths import DATA_DIR, CONFIG_DIR


def main(args):
  r"""Inserts new documents in the collection.
  """
  assert isfile(args.documents), f'Documents file {args.documents} does not exist'
  config = load_config(args.config)
  collection_name = get_my_collection_name(config.github_username)

  # Load raw documents
  # TODO: Add column descriptions here
  raw = pd.read_csv(args.docs_csv)

  documents = []
  # ===========================
  # FILL ME OUT
  # Prepare the documents to be inserted into the vector db
  # Example document:
  # {
  #   "embeddings": {
  #     "values": [0.1, 0.2, 0.3, 0.4, 0.5],
  #     "dimensionality": 5,
  #   }, # single vector document
  #   "metadata": {
  #     "label1": "0",
  #     "label2": "1",
  #   }
  # }
  # Metadata is optional if you have additional attributes. 
  # Please see docs here: https://docs.starpoint.ai/create-documents 
  # ===========================

  result = insert_documents(args.starpoint_api_key, collection_name, documents)
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