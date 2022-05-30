import torch
import numpy as np
import pandas as pd
from os.path import join
from src.paths import DATA_DIR


def main():
  rs = np.random.RandomState(42)
  en_dir = join(DATA_DIR, 'en')
  es_dir = join(DATA_DIR, 'es')
  out_dir = join(DATA_DIR, 'stream')

  en_csv, en_emb = load_dev_and_test(en_dir)
  es_csv, es_emb = load_train(es_dir, max_size=len(en_csv))
  portions = [0, 0, 0.2, 0.4, 0.6, 0.8, 0.8, 0.8]

  for i in range(8):
    size = 1000
    frac = portions[i]
    en_csv_part = en_csv[i*1000:(i+1)*1000]
    es_csv_part = es_csv[i*1000:(i+1)*1000]
    csv_i = pd.concat([
      en_csv_part[:int(round(1 - frac, 2) * size)], 
      es_csv_part[:int(frac * size)]
    ])
    csv_i = csv_i.reset_index(drop=True)

    en_emb_part = en_emb[i*1000:(i+1)*1000]
    es_emb_part = es_emb[i*1000:(i+1)*1000]
    emb_i = torch.cat([
      en_emb_part[:int(round(1 - frac, 2) * size)], 
      es_emb_part[:int(frac * size)]
    ])

    indices = np.arange(1000)
    rs.shuffle(indices)
    indices = indices.tolist()

    csv_i = csv_i.loc[indices]
    emb_i = emb_i[indices]

    csv_i.to_csv(join(out_dir, f'stream{i+1}.csv'), index=False)
    torch.save(emb_i, join(out_dir, f'stream{i+1}.pt'))


def load_dev_and_test(dir):
  dev_csv = pd.read_csv(join(dir, 'dev.csv'))
  test_csv = pd.read_csv(join(dir, 'test.csv'))
  dev_emb = torch.load(join(dir, 'dev.pt'))
  test_emb = torch.load(join(dir, 'test.pt'))

  csv = pd.concat([dev_csv, test_csv])
  emb = torch.cat([dev_emb, test_emb], dim=0)
  csv = csv.reset_index(drop=True)

  return csv, emb


def load_train(dir, max_size):
  train_emb = torch.load(join(dir, 'train.pt'))
  train_csv = pd.read_csv(join(dir, 'train.csv'))

  csv = train_csv.loc[:max_size]
  emb = train_emb[:max_size]

  return csv, emb


if __name__ == "__main__":
  main()