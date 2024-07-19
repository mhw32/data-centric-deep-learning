import re
import torch
import numpy as np
import pandas as pd
from os.path import join
from torch.utils.data import Dataset

from src.consts import DATA_DIR


class ProductReviewEmbeddings(Dataset):
  r"""RoBERTa embeddings of customer reviews. Embeddings are precomputed 
  and saved to disk. This class does not compute embeddings live.

  Argument
  --------
  split (str): the dataset portion
    Options - train | dev | test | unlabeled | * 
    If unlabeled, the __getitem__ function will set `label` to -1.
  """

  def __init__(self, split='train'):
    super().__init__()
    self.data = pd.read_csv(join(DATA_DIR, f'{split}.csv'))
    self.embedding = torch.load(join(DATA_DIR, f'{split}.pt'))
    self.split = split

  def __getitem__(self, index):
    inputs = self.embedding[index].float()
    label = int(self.data.iloc[index].label)
    return inputs, label

  def __len__(self):
    return len(self.data)

