import torch
from os.path import join
from testing.system import DigitClassifierSystem, MNISTDataModule
from testing.regression import build_regression_test
from testing.paths import IMAGE_DIR


def main(args):
  '''Builds a regression test set from a model checkpoint.
  '''
  system = DigitClassifierSystem.load_from_checkpoint(args.ckpt)
  dm = MNISTDataModule(system.config)
  loader = dm.val_dataloader()

  # Builds the regression examples
  images, labels = build_regression_test(system, loader)

  save_path = join(IMAGE_DIR, 'regression/test-data.pt')
  os.makedirs(os.path.dirname(save_path), exist_ok=True)
  torch.save({'images': images, 'labels': labels}, save_path)

  print(f'Saved to {save_path}.')


if __name__ == "__main__":
  # Given the checkpoint file for a trained system, build a regression
  # file and save it to disk.
  import os
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('ckpt', type=str, help='path to trained checkpoint file')
  args = parser.parse_args()

  main(args)