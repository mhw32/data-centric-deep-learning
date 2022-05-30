from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from src.monitor import MonitoringSystem
from src.dataset import ProductReviewStream, ProductReviewEmbeddings
from src.systems import SentimentClassifierSystem


def main(args):
  system = SentimentClassifierSystem.load_from_checkpoint(args.ckpt)
  tr_ds = ProductReviewEmbeddings(lang=system.config.system.data.lang, split='train')
  tr_dl = DataLoader(tr_ds, batch_size=128, shuffle=False, num_workers=4)
  tr_vocab = tr_ds.get_vocab()
  tr_probs = get_probs(system, tr_dl)

  monitor = None
  # ============================
  # FILL ME OUT
  # 
  # Initialize the `MonitoringSystem` using the vocabulary
  # and predicted probabilities.
  # 
  # Pseudocode:
  # --
  # monitor = MonitoringSystem(...)
  #
  # Type:
  # --
  # monitor: MonitoringSystem
  monitor = MonitoringSystem(tr_vocab, tr_probs)
  # ============================

  for index in range(1, 9):
    te_ds = ProductReviewStream(index)
    te_dl = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=4)
    te_vocab = te_ds.get_vocab()
    te_probs = get_probs(system, te_dl)

    results = None
    # ============================
    # FILL ME OUT
    # 
    # Pass `te_vocab` and `te_probs` to the monitor to 
    # compute monitored results.
    # 
    # Pseudocode:
    # --
    # Call `monitor.monitor` to compute results.
    # 
    # Type:
    # --
    # results: Dict[str, Any] - results from monitoring
    #   Expected keys:
    #     - ks_score: p-value from two-sample KS test
    #     - hist_score: intersection score between histograms
    #     - outlier_score: perc of vocabulary that is new
    results = monitor.monitor(te_vocab, te_probs)
    # ============================

    if results is not None:
      print('==========================')
      print(f'STREAM ({index} out of 8)')
      print('==========================')
      print(f'KS test p-value: {results["ks_score"]:.3f}')
      print(f'Histogram intersection: {results["hist_score"]:.3f}')
      print(f'OOD Vocab %: {results["outlier_score"]*100:.2f}')
      print('')  # new line


def get_probs(system, loader):
  trainer = Trainer()
  probs = trainer.predict(system, dataloaders=loader)
  return probs


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('ckpt', type=str, help='path to checkpoint file')
  args = parser.parse_args()
  main(args)