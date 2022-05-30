"""Get foundation model embeddings using RoBERTa."""

import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from datasets import load_dataset


class ProductReviews(Dataset):

  def __init__(self, lang='en', split='train'):
    super().__init__()
    assert lang in ['en', 'es', 'de'], f"Language {lang} not supported."
    assert split in ['train', 'dev', 'test'], f"Split {split} not supported."
    if split == 'dev': split = 'validation'

    dataset = load_dataset('amazon_reviews_multi', lang, split=split)

    self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    self.dataset = self.prepare(dataset)
    self.split = split
    self.lang = lang

  def prepare(self, dataset):
    processed = []
    for i in range(len(dataset)):
      row = dataset.__getitem__(i)
      if row['stars'] != 3:  # ignore these ambiguous ones
        processed.append({
          'data': row['review_body'],
          'label': 0 if row['stars'] < 3 else 1,
        })
    return processed

  def __getitem__(self, index):
    row = self.dataset[index]
    review = str(row['data'])
    label = int(row['label'])

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
    return len(self.dataset)


def main(lang='en', batch_size=16, device=0):
  device = torch.device(f'cuda:{device}')

  train_dataset = ProductReviews(lang=lang, split='train')
  dev_dataset = ProductReviews(lang=lang, split='dev')
  test_dataset = ProductReviews(lang=lang, split='test')

  pretrained = RobertaModel.from_pretrained('roberta-base')
  pretrained = pretrained.to(device)

  @torch.no_grad()
  def precompute_features(dataset, batch_size = 4):
    loader = DataLoader(dataset, batch_size = batch_size, shuffle = False, num_workers = 4)
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

  train_features = precompute_features(train_dataset, batch_size = batch_size)
  dev_features = precompute_features(dev_dataset, batch_size = batch_size)
  test_features = precompute_features(test_dataset, batch_size = 16)

  script_dir = os.path.dirname(__file__)
  data_dir = os.path.join(script_dir, '../data')
  emb_dir = os.path.join(data_dir, f'emb/{lang}')
  os.makedirs(emb_dir, exist_ok=True)

  torch.save(train_features, os.path.join(emb_dir, f'train.pt'))
  torch.save(dev_features, os.path.join(emb_dir, f'dev.pt'))
  torch.save(test_features, os.path.join(emb_dir, f'test.pt'))


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--lang', type=str, default='en', choices=['en', 'es', 'de'])
  parser.add_argument('--device', type=int, default=0)
  parser.add_argument('--batch-size', type=int, default=16)
  args = parser.parse_args()
  main(lang=args.lang, batch_size=args.batch_size, device=args.device)

