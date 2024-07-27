import pandas as pd
from typing import List
from glob import glob
from os.path import join, basename, splitext
from .paths import DATA_DIR


def load_documents() -> pd.DataFrame:
  r"""Load in all documents. Use filename as doc ids.
  """
  doc_dir = join(DATA_DIR, 'documents')
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
  # ===========================
  # FILL ME OUT
  # ===========================
  assert len(chunks) > 0, f"Remember to complete `chunk_document`."
  return chunks
