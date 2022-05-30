"""Get foundation model embeddings using RoBERTa."""

import os
import torch
import pandas as pd
from os.path import join
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
from datasets import load_dataset
from src.paths import DATA_DIR


def prepare(dataset):
  processed_data = []
  processed_label = []
  
  for i in range(len(dataset)):
    row = dataset.__getitem__(i)
    if row['stars'] != 3:  # ignore these ambiguous ones
      processed_data.append(row['review_body'])
      processed_label.append(0 if row['stars'] < 3 else 1,)

  processed = {'review': processed_data, 'label': processed_label}
  processed = pd.DataFrame.from_dict(processed)

  return processed


def main(lang='en'):
  train_ds = load_dataset('amazon_reviews_multi', lang, split='train')
  dev_ds = load_dataset('amazon_reviews_multi', lang, split='validation')
  test_ds = load_dataset('amazon_reviews_multi', lang, split='test')

  train_df = prepare(train_ds)
  dev_df = prepare(dev_ds)
  test_df = prepare(test_ds)

  train_df.to_csv(join(DATA_DIR, lang, 'train.csv'), index=False)
  dev_df.to_csv(join(DATA_DIR, lang, 'dev.csv'), index=False)
  test_df.to_csv(join(DATA_DIR, lang, 'test.csv'), index=False)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--lang', type=str, default='en', choices=['en', 'es', 'de'])
  parser.add_argument('--device', type=int, default=0)
  parser.add_argument('--batch-size', type=int, default=16)
  args = parser.parse_args()
  main(lang=args.lang)

