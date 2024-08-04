from tqdm import tqdm
from os.path import join
from sentence_transformers import SentenceTransformer

from rag.llm import embedding_name_to_dim
from rag.vector import get_my_collection_name, insert_documents
from rag.dataset import load_documents
from rag.paths import DATA_DIR


def main(args):
  r"""Inserts new documents in the collection.
  """
  collection_name = get_my_collection_name(
    env['GITHUB_USERNAME'],
    embedding=args.embedding, 
    hyde=args.hyde,
  )

  # Load raw documents as a Pandas Dataframe with two columns
  # - doc_id: Document ID 
  # - text: Content for the document
  raw = load_documents(override_doc_dir=args.doc_dir)
  print(f'Found {len(raw)} documents to upload.')

  # Use the embedding model to embed docs
  embedding_dim = embedding_name_to_dim(args.embedding)
  embedding_model = SentenceTransformer(args.embedding)
  print(f'Loaded the {args.embedding} model.')

  documents = []
  for i in tqdm(range(len(raw)), desc='Inserting into db'):
    doc = ""
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
    # TODO
    # ===========================
    assert len(doc) > 0, f"Did you complete the code in `insert_docs.py`?"
    documents.append(doc)

  assert len(documents) > 0, f"Please remember to append to the documents array"
  
  print(f'Inserting documents into Starpoint collection {collection_name}')
  insert_documents(args.starpoint_api_key, collection_name, documents)
  print(f'Done. {len(documents)} inserted.')


if __name__ == "__main__":
  from os import environ as env
  from dotenv import load_dotenv
  load_dotenv()

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--embedding', type=str, default='all-MiniLM-L6-v2', help='Embedding to use (default: all-MiniLM-L6-v2)')
  parser.add_argument('--hyde', action='store_true', default=False, help='Use hyde embeddings (default: False)')
  parser.add_argument('--doc-dir', type=str, default=join(DATA_DIR, 'documents/summer'), help='Document directory')
  parser.add_argument('--starpoint-api-key', type=str, default=env['STARPOINT_API_KEY'], help='Starpoint API key')
  args = parser.parse_args()

  main(args)