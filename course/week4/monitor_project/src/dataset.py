import torch
import pandas as pd
from os.path import join
from torch.utils.data import Dataset
from collections import defaultdict
from src.paths import DATA_DIR


class ProductReviewEmbeddings(Dataset):
  r"""RoBERTa embeddings of customer reviews. Embeddings are precomputed 
  and saved to disk. This class does not compute embeddings live.

  Argument
  --------
  lang (str): the language
    Options - en | es | de
  split (str): the dataset portion
    Options - train | dev | test | * 
  """
  def __init__(self, lang='en', split='train'):
    super().__init__()
    self.data = pd.read_csv(join(DATA_DIR, lang, f'{split}.csv'))
    self.embedding = torch.load(join(DATA_DIR, lang, f'{split}.pt'))
    self.split = split
    self.lang = lang

  def get_vocab(self):
    vocab = defaultdict(lambda: 0)
    # ===============================
    # FILL ME OUT
    # 
    # Compute a map between word to count: number of times the 
    # word shows up in the dataset.
    # 
    # Pseudocode:
    # --
    # loop through `self.data.review`
    #   split review into tokens
    #   update vocab with each token
    # 
    # Type:
    # --
    # vocab: dict[str, int]
    # 
    # Notes:
    # --
    # Convert tokens to lowercase when updating vocab.
    for review in self.data.review:
      tokens = review.split()
      for token in tokens:
        vocab[token.lower()] += 1
    # ===============================
    return dict(vocab)

  def __getitem__(self, index):
    label = self.data.iloc[index].label
    output = {
      'embedding': self.embedding[index].float(),
      'label': int(label),
    }
    return output

  def __len__(self):
    return len(self.data)


class ProductReviewStream(Dataset):
  r"""Simulates a stream of customer reviews. Embeddings are precomputed 
  and saved to disk. This class does not compute embeddings live. No labels
  will be provided here.

  Argument:
  --------
  index (int): stream index 
    Options - 1 to 9
  """
  def __init__(self, index):
    super().__init__()
    assert index in range(1, 9), f"Invalid index: {index}"
    self.data = pd.read_csv(join(DATA_DIR, 'stream', f'stream{index}.csv'))
    self.embedding = torch.load(join(DATA_DIR, 'stream', f'stream{index}.pt'))

  def get_vocab(self):
    # `defaultdict` can be a helpful utility
    vocab = defaultdict(lambda: 0)
    # ===============================
    # FILL ME OUT
    # 
    # Copy your implementation from `ProductReviewEmbeddings.get_vocab`.
    for review in self.data.review:
      tokens = review.split()
      for token in tokens:
        vocab[token.lower()] += 1
    # ===============================
    return dict(vocab)

  def __getitem__(self, index):
    output = {
      'embedding': self.embedding[index].float(),
    }
    return output

  def __len__(self):
    return len(self.data)