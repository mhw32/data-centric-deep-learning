from os import makedirs
from os.path import join
from rag.vector import get_my_collection_name, create_collection
from rag.llm import embedding_name_to_dim
from rag.utils import to_json
from rag.paths import DATA_DIR


def main(args):
  r"""Create a Starpoint collection. 
  :note: This will save a file to data/collections that remembers your collection name and ID
  """
  collection_name = get_my_collection_name(env['GITHUB_USERNAME'], embedding=args.embedding, hyde=args.hyde)
  collection = create_collection(
    args.starpoint_api_key, 
    collection_name=collection_name, 
    dimensionality=embedding_name_to_dim(args.embedding),
  )
  # Save info to the collections dir
  makedirs(join(DATA_DIR, 'collections'), exist_ok=True)
  collection_file = join(DATA_DIR, 'collections', f'{collection_name}.json')
  to_json(collection, collection_file)

  print(f'Collection {collection_name} created. Info saved to {collection_file}')


if __name__ == "__main__":
  from os import environ as env
  from dotenv import load_dotenv
  load_dotenv()

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--embedding', type=str, default='all-MiniLM-L6-v2', help='Embedding to use (default: all-MiniLM-L6-v2)')
  parser.add_argument('--hyde', action='store_true', default=False, help='Use hyde embeddings (default: False)')
  parser.add_argument('--starpoint-api-key', type=str, default=env['STARPOINT_API_KEY'], help='Starpoint API key')
  args = parser.parse_args()

  main(args)