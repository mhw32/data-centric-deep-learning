"""Get foundation model embeddings using RoBERTa."""

import os
import torch
import jsonlines
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer

from src.consts import DATA_DIR


class ProductReviews(Dataset):
  r"""Customer reviews of beauty products.

  Argument
  --------
  category (str): the product category
    Optiona - luxury-beauty | musical-instruments | video-games
  split (str): the dataset portion
    Options - train | dev | test
  """

  def __init__(self, root, category='luxury-beauty', split='train'):
    super().__init__()
    assert split in ['train', 'dev', 'test'], f"Split {split} not supported."

    # load a category and a split
    self.data = pd.read_csv(os.path.join(root, category, f'{split}.csv'))

    # initialize the RoBERTa which converts natural language to tokens
    self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    self.split = split
    self.category = category

  def __getitem__(self, index):
    row = self.data.iloc[index]

    review = str(row.review)
    label = int(row.label)

    tokenized = self.tokenizer(
      review,
      truncation=True, 
      padding='max_length',
      pad_to_max_length=True, 
      return_attention_mask=True,
      return_tensors='pt',
    )
    output = {
      'input_ids': tokenized['input_ids'].squeeze(0),
      'attention_mask': tokenized['attention_mask'].squeeze(0),
      'label': label,
    }
    return output

  def __len__(self):
    return len(self.data)


def get_product_embeddings(category='luxury-beauty'):
  device = torch.device('cuda:3')

  train_dataset = ProductReviews(DATA_DIR, category=category, split='train')
  dev_dataset = ProductReviews(DATA_DIR, category=category, split='dev')
  test_dataset = ProductReviews(DATA_DIR, category=category, split='test')

  pretrained = RobertaModel.from_pretrained('roberta-base')
  pretrained = pretrained.to(device)

  @torch.no_grad()
  def precompute_features(dataset, batch_size = 4):
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False)
    features = []
    pbar = tqdm(total=len(loader))
    for batch in loader:
      outputs = pretrained(
        input_ids = batch['input_ids'].to(device),
        attention_mask = batch['attention_mask'].to(device))
      feature = outputs['pooler_output']
      features.append(feature.detach().cpu())
      pbar.update()
    pbar.close()
    features = torch.cat(features, dim=0)
    return features

  train_features = precompute_features(train_dataset)
  dev_features = precompute_features(dev_dataset)
  test_features = precompute_features(test_dataset)

  torch.save(train_features, os.path.join(DATA_DIR, category, f'train.pt'))
  torch.save(dev_features, os.path.join(DATA_DIR, category, f'dev.pt'))
  torch.save(test_features, os.path.join(DATA_DIR, category, f'test.pt'))


def main():  
  get_product_embeddings(category='video-games')
  get_product_embeddings(category='musical-instruments')
  get_product_embeddings(category='luxury-beauty')


if __name__ == "__main__":
  main()
