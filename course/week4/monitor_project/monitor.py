from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
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
  results = []

  for index in range(1, 9):
    te_ds = ProductReviewStream(index)
    te_dl = DataLoader(te_ds, batch_size=128, shuffle=False, num_workers=4)
    te_vocab = te_ds.get_vocab()
    te_probs = get_probs(system, te_dl)

    cur_results = None
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
    # cur_results: Dict[str, Any] - results from monitoring
    cur_results = monitor.monitor(te_vocab, te_probs)
    # ============================
    results.append(cur_results)

  return results


def get_probs(system, loader):
  trainer = Trainer()
  probs = trainer.predict(dataloaders=loader)
  return probs


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('ckpt', type=str, help='path to checkpoint file')
  args = parser.parse_args()
  main(args)