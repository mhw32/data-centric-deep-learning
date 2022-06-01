from os.path import join
from pprint import pprint
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from src.systems import RobustSentimentSystem, ReviewDataModule
from src.utils import load_config
from src.paths import CONFIG_DIR


def main():
  config = load_config(join(CONFIG_DIR, f'dro.json'))
  dm = ReviewDataModule(config)
  breakpoint()
  system = RobustSentimentSystem(config)

  checkpoint_callback = ModelCheckpoint(
    dirpath = config.system.save_dir,
    save_last = True,
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
  main()
