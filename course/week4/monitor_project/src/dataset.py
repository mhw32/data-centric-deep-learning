import torch
import numpy as np
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
    Options - en | es | mix
  split (str): the dataset portion
    Options - train | dev | test | * 
  """
  def __init__(self, lang = 'en', split = 'train', weights = None):
    super().__init__()
    self.data = pd.read_csv(join(DATA_DIR, lang, f'{split}.csv'))
    self.embedding = torch.load(join(DATA_DIR, lang, f'{split}.pt'))
    if weights is None:
      weights = torch.ones(self.embedding.size(0))
    assert len(weights) == len(self.embedding)
    self.weights = weights
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
    # ===============================
    return dict(vocab)

  def get_labels(self):
    # return labels as a torch.LongTensor
    label = np.asarray(self.data.label)
    label = torch.from_numpy(label).long()
    return label

  def __getitem__(self, index):
    row = self.data.iloc[index]
    output = {
      'embedding': self.embedding[index].float(),
      'label': int(row.label),
      # useful for DRO
      'group': int(row.group if 'group' in row else -1),
      # useful for JTT
      'weight': self.weights[index].item(),
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
    # Copy your implementation of `get_vocab` from 
    # the `ProductReviewEmbeddings` class here.
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
    # ===============================
    return dict(vocab)

  def __getitem__(self, index):
    output = {
      'embedding': self.embedding[index].float(),
    }
    return output

  def __len__(self):
    return len(self.data)