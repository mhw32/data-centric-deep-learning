"""
Integration tests must pass for every new model version. These 
represent core functionality that a model must pass to be deployed
in real world applications.
"""
import os
import pandas as pd
from glob import glob
from os.path import join
from PIL import Image
from torchvision import transforms

from torch.utils.data import Dataset, DataLoader
from src.tests.base import BaseTest


class MNISTIntegrationTest(BaseTest):
  """An integration test include a set of examples that a model 
  must pass before being deployed in practice. For this project,
  we include a set of 10 handwritten digits (provided by you!) that
  the model must correctly classify.
  """

  def __init__(self):
    super().__init__()
    paths, labels = [], []

    # Store the paths of all processed images in the `paths` list.
    # Use the `labels.csv` to fill out the `labels` list with the 
    # corresponding labels for each path in `paths`.
    test_dir = join(self.root, 'integration')
    labels_df = pd.read_csv(join(test_dir, 'labels.csv'))
    labels_dict = dict(zip(labels_df.path, labels_df.label))
    paths = glob(join(test_dir, 'digits-processed', '*.png'))
    for path in paths:
      name = os.path.basename(path)
      label = labels_dict[name]
      labels.append(label)

    self.paths = paths
    self.labels = labels

  def get_dataloader(self, batch_size = 10):
    # returns a data loader
    dataset = MNISTIntegrationDataset(self.paths, self.labels, 
      transform = transforms.ToTensor())
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader

  def test(self, trainer, system):
    # ================================
    # FILL ME OUT
    #
    # Create a dataloader, and pass it to the trainer and call `test`.
    # Our solution is two lines of code.
    #
    # Pseudocode:
    # --
    # loader = ...
    # pass loader to trainer and call test
    #
    # Notes:
    # --
    # Nothing to return here
    pass  # remove me
    # ================================


class MNISTIntegrationDataset(Dataset):
  """
  A dataset for integration tests on MNIST.

  NOTE: You will not need to edit this.
  """
  def __init__(self, paths, labels, transform=None):
    super().__init__()
    self.paths = paths
    self.labels = labels
    self.transform = transform

  def __getitem__(self, index):
    path = self.paths[index]
    img = Image.open(path)

    if self.transform is not None:
      img = self.transform(img)

    return img, self.labels[index]

  def __len__(self):
    return len(self.paths)


if __name__ == "__main__":
  dataset = MNISTIntegrationTest()
  num_examples = len(dataset.paths)
  print(f'# of examples: {num_examples}')
