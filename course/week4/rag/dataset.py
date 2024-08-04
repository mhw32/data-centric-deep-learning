import pandas as pd
from typing import List, Optional
from glob import glob
from os.path import join, basename, splitext
from .paths import DATA_DIR


def load_documents(override_doc_dir: Optional[str] = None) -> pd.DataFrame:
  r"""Load in all documents. Use filename as doc ids.
  """
  if override_doc_dir:
    doc_dir = override_doc_dir
  else:
    doc_dir = join(DATA_DIR, 'documents/summer')
  doc_files = glob(join(doc_dir, '*.md'))  # find all markdown files

  doc_ids, texts = [], []
  for doc_file in doc_files:
    with open(doc_file, 'r') as fp:
      doc = fp.read()
    doc_id, _ = splitext(basename(doc_file))
    doc_ids.append(doc_id)
    texts.append(doc)

  dataset = pd.DataFrame({'doc_id': doc_ids, 'text': texts})
  return dataset


def chunk_document(text: str) -> List[str]:
  r"""Split a document into chunks for question generation.
  Do this by splitting between H2 headers (`##`).
  """
  chunks: List[str] = []
  # In generating questions, we typically do not want to provide the whole 
  # document. We would prefer to break the document into semantic chunks to
  # be able to generate a question for each chunk.
  chunks = [chunk.strip() for chunk in text.split('##')]
  return chunks
