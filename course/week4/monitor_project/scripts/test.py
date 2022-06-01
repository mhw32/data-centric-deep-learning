from pprint import pprint
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.dataset import ProductReviewEmbeddings
from src.systems import SentimentClassifierSystem
from src.paths import LOG_DIR


def main(args):
  ds = ProductReviewEmbeddings(lang=args.lang, split='test')
  dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4)

  system = SentimentClassifierSystem.load_from_checkpoint(args.ckpt)
  trainer = Trainer(logger = TensorBoardLogger(save_dir = LOG_DIR))
  trainer.test(system, dataloaders=dl)

  results = system.test_results
  pprint(results)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('ckpt', type=str, help='path to checkpoint')
  parser.add_argument('--lang', type=str, default='en', choices=['en', 'es', 'de'])
  args = parser.parse_args()
  main(args)
