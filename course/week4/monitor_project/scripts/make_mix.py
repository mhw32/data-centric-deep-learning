import torch
import numpy as np
import pandas as pd
from os.path import join
from src.paths import DATA_DIR


def main():
  rs = np.random.RandomState(42)
  en_dir = join(DATA_DIR, 'en')
  es_dir = join(DATA_DIR, 'es')
  out_dir = join(DATA_DIR, 'mix')

  for split in ['train', 'dev', 'test']:
    en_csv, en_emb = load_split(en_dir, split=split, rs=rs)
    es_csv, es_emb = load_split(es_dir, split=split, rs=rs)

    size = len(en_emb)
    en_size = int(0.90 * size)
    es_size = size - en_size
    group = np.concatenate([np.ones(es_size), np.zeros(en_size)])

    csv = pd.concat([es_csv.iloc[:es_size], en_csv.iloc[:en_size]])
    emb = torch.cat([es_emb[:es_size], en_emb[:en_size]])
    csv = csv.reset_index(drop=True)
    csv['group'] = group.astype(int)

    indices = np.arange(len(csv))
    rs.shuffle(indices)

    csv = csv.iloc[indices]
    emb = emb[indices]
    csv = csv.reset_index(drop=True)

    csv.to_csv(join(out_dir, f'{split}.csv'), index=False)
    torch.save(emb, join(out_dir, f'{split}.pt'))


def load_split(dir, split='train', rs=None):
  if rs is None:
    rs = np.random.RandomState(42)

  emb = torch.load(join(dir, f'{split}.pt'))
  csv = pd.read_csv(join(dir, f'{split}.csv'))

  indices = np.arange(len(csv))
  rs.shuffle(indices)

  csv = csv.loc[indices]
  emb = emb[indices]

  return csv, emb


if __name__ == "__main__":
  main()
