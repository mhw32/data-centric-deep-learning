from os import remove
from os.path import join, isfile
from rag.vector import get_my_collection_name, create_collection, delete_collection
from rag.paths import DATA_DIR
from rag.utils import from_json


def main(args):
  r"""Create a Starpoint collection
  """
  collection_name = get_my_collection_name(args.github_username)
  collection_file = join(DATA_DIR, 'collections', f'{collection_name}.json')
  collection = from_json(collection_file)

  # Tell starpoint to delete the collection
  success = delete_collection(args.starpoint_api_key, collection['id'])

  if isfile(collection_file):  # Delete the file as well
    remove(collection_file)


if __name__ == "__main__":
  from os import environ as env
  from dotenv import load_dotenv
  load_dotenv()

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('github_username', type=str, help='Your github username')
  parser.add_argument('--starpoint-api-key', type=str, default=env['STARPOINT_API_KEY'], help='Starpoint API key')
  args = parser.parse_args()

  main(args)