import torch
import numpy as np
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from monitor.metrics import MonitoringSystem
from monitor.dataset import ProductReviewStream, ProductReviewEmbeddings
from monitor.systems import SentimentClassifierSystem
from monitor.paths import LOG_DIR


def main(args):
  rs = np.random.RandomState(42)
  system = SentimentClassifierSystem.load_from_checkpoint(args.checkpoint)
  tr_ds = ProductReviewEmbeddings(lang=system.config.data.lang, split='train')
  tr_dl = DataLoader(tr_ds, batch_size=128, shuffle=False, num_workers=4)
  tr_vocab = tr_ds.get_vocab()

  tr_probs = get_probs(system, tr_dl)
  tr_labels = tr_ds.get_labels()

  # we don't want to use all the training set as it much larger than
  # our stream datasets. We randomly pick 1,000.
  tr_probs, tr_labels = create_sample(tr_probs, tr_labels, 1000, rs)

  # initialize the `MonitoringSystem` using the vocabulary
  # and predicted probabilities.
  monitor = MonitoringSystem(tr_vocab, tr_probs, tr_labels)

  for index in range(1, 9):
    te_ds = ProductReviewStream(index)
    te_dl = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=4)
    te_vocab = te_ds.get_vocab()
    te_probs = get_probs(system, te_dl)

    results = None

    # Compute monitored results.
    # 
    # results: Dict[str, Any] - results from monitoring
    #   keys:
    #   --
    #   ks_score: p-value from two-sample KS test
    #   hist_score: intersection score between histograms
    #   outlier_score: perc of vocabulary that is new
    results = monitor.monitor(te_vocab, te_probs)

    if results is not None:
      print('\n==========================')
      print(f'STREAM ({index} out of 8)')
      print('==========================')
      print(f'KS test p-value: {results["ks_score"]:.4f}')
      print(f'KS test p-value(using uncalibrated score): {results["ks_score_uncalibrated_p"]:.4f}')
      print(f'Histogram intersection: {results["hist_score"]:.4f}')
      print(f'Histogram intersection(using uncalibrated score): {results["hist_score_uncalibrated_p"]:.4f}')
      print(f'OOD Vocab %: {results["outlier_score"]*100:.2f}')
      print('')  # new line


def get_probs(system, loader):
  trainer = Trainer(logger = TensorBoardLogger(save_dir=LOG_DIR))
  probs = trainer.predict(system, dataloaders=loader)
  return torch.cat(probs, dim=0).squeeze(1)


def create_sample(probs, labels, size, rs):
  indices = np.arange(len(probs))
  indices = rs.choice(indices, size=size, replace=False)
  probs = probs[indices]
  labels = labels[indices]
  return probs, labels


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('checkpoint', type=str, help='path to checkpoint file')
  args = parser.parse_args()
  main(args)
