from os import remove
from os.path import join, isfile
from rag.vector import get_my_collection_name, delete_collection
from rag.utils import from_json
from rag.paths import DATA_DIR


def main(args):
  r"""Deletes a Starpoint collection
  :note: This will find the collection ID using a file in data/collections.
  """
  collection_name = get_my_collection_name(env['GITHUB_USERNAME'], embedding=args.embedding, hyde=args.hyde)
  collection_file = join(DATA_DIR, 'collections', f'{collection_name}.json')
  assert isfile(collection_file), f'Collection file {collection_file} not found.'
  collection = from_json(collection_file)

  # Tell starpoint to delete the collection
  success = delete_collection(args.starpoint_api_key, collection['id'])
  if not success:
    raise Exception(f'Failed to delete collection. Please check your collection ID')

  if success and isfile(collection_file):  # Delete the file as well
    remove(collection_file)


if __name__ == "__main__":
  from os import environ as env
  from dotenv import load_dotenv
  load_dotenv()

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('github_username', type=str, help='Your github username')
  parser.add_argument('--embedding', type=str, help='Embedding to use (default: all-MiniLM-L6-v2)')
  parser.add_argument('--hyde', action='store_true', default=False, help='Use hyde embeddings (default: False)')
  parser.add_argument('--starpoint-api-key', type=str, default=env['STARPOINT_API_KEY'], help='Starpoint API key')
  args = parser.parse_args()

  main(args)
