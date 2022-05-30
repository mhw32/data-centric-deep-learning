from os.path import join
from pprint import pprint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.systems import SentimentClassifierSystem, ReviewDataModule
from src.utils import load_config
from src.paths import CONFIG_DIR


def main(args):
  config = load_config(join(CONFIG_DIR, f'train_{args.lang}.json'))
  dm = ReviewDataModule(config)
  system = SentimentClassifierSystem(config)

  checkpoint_callback = ModelCheckpoint(
    dirpath = config.system.save_dir,
    monitor = 'dev_loss',
    mode = 'min',
    save_top_k = 1,
    verbose = True,
  )

  trainer = Trainer(
    max_epochs = config.system.optimizer.max_epochs,
    callbacks = [checkpoint_callback])

  trainer.fit(system, dm)
  trainer.test(system, dm, ckpt_path = 'best')

  results = system.test_results
  pprint(results)


if __name__ == "__main__":
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--lang', type=str, default='en', choices=['en', 'es', 'de'])
  args = parser.parse_args()
  main(args)
