r"""A PyTorch Lightning system for training MNIST."""

import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader

import pytorch_lightning as pl
from torchvision import datasets, transforms


class MNISTDataModule(pl.LightningDataModule):
  r"""Data module wrapper around MNIST datasets."""

  def __init__(self, config):
    super().__init__()

    # Load the training and test splits of MNIST 
    # using `torchvision.datasets`. See
    # https://pytorch.org/vision/stable/datasets.html 
    train_dataset = datasets.MNIST(
      config.system.data.root,
      download = True,
      train = True,
      transform = transforms.ToTensor())

    test_dataset = datasets.MNIST(
      config.system.data.root,
      download = True,
      train = False,
      transform = transforms.ToTensor())

    assert len(train_dataset) == 60000, \
      f"Unexpected training data size: {len(train_dataset)}"

    assert len(test_dataset) == 10000, \
      f"Unexpected test data size: {len(test_dataset)}"

    # Split the train dataset into a train and dev set.
    # Keep 80% of the `train_dataset` as training and use the
    # rest as a dev set. 
    dev_size = int(len(train_dataset) * 0.2)
    indices = torch.randperm(len(train_dataset)).tolist()
    _train_dataset = Subset(train_dataset, indices[:-dev_size])
    dev_dataset = Subset(train_dataset, indices[-dev_size:])
    train_dataset = _train_dataset  # overwrite object

    assert len(train_dataset) == 48000, \
      f"Unexpected train data size: {len(train_dataset)}"
    assert len(dev_dataset) == 12000, \
      f"Unexpected dev data size: {len(dev_dataset)}"
    assert len(test_dataset) == 10000, \
      f"Unexpected test data size: {len(test_dataset)}"

    self.train_dataset = train_dataset
    self.dev_dataset = dev_dataset
    self.test_dataset = test_dataset
    self.batch_size = config.system.optimizer.batch_size

  def train_dataloader(self):
    # Create a dataloader for train dataset. 
    # Set `shuffle=True` for the training dataset.
    return DataLoader(self.train_dataset, batch_size = self.batch_size, 
      shuffle = True)

  def val_dataloader(self):
    return DataLoader(self.dev_dataset, batch_size = self.batch_size)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size = self.batch_size)


class DigitClassifierSystem(pl.LightningModule):
  """Remember PyTorch Lightning from the DL Refresher in Week 1?

  A Pytorch Lightning system to train a model to classify handwritten digits 
  using the MNIST dataset.

  Arguments
  ---------
  config (dotmap.DotMap): a configuration file with hyperparameters.
    See config.py for an example.
  """
  def __init__(self, config):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    # make directory to store data
    os.makedirs(config.system.data.root, exist_ok = True)

    # load model
    self.model = self.get_model()

    # We will overwrite this once we run `test()`
    self.test_results = {}

  def get_model(self):
    if self.config.system.model.name == 'linear':
      model = nn.Linear(784, 10)

    elif self.config.system.model.name == 'mlp':
      model = nn.Sequential(
        nn.Linear(784, self.config.system.model.width),
        nn.ReLU(),
        nn.Linear(self.config.system.model.width, 10)
      )

    else:
      raise Exception(f"Model {self.config.system.model.name} not supported.")

    return model

  def forward(self, image):
    image = image.view(image.size(0), -1)  # flatten the image
    logits = self.model(image)
    return logits
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.config.system.optimizer.lr)
    return optimizer

  def _common_step(self, batch, batch_idx):
    """
    Arguments
    ---------
    images (torch.Tensor): transformed images
      shape: batch_size x 1 x 28 x 28
    labels (torch.LongTensor): labels from 0 -> 9
      shape: batch_size
    """
    images, labels = batch

    # Compute loss using inputs and labels
    logits = self.forward(images)
    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
      # Compute accuracy using the logits and labels
      preds = torch.argmax(logits, dim=1)
      num_correct = torch.sum(preds == labels)
      num_total = labels.size(0)
      accuracy = num_correct / float(num_total)

    return loss, accuracy

  def training_step(self, train_batch, batch_idx):
    loss, acc = self._common_step(train_batch, batch_idx)
    self.log_dict({'train_loss': loss, 'train_acc': acc},
      on_step=True, on_epoch=False, prog_bar=True, logger=True)
    return loss

  def validation_step(self, dev_batch, batch_idx):
    loss, acc = self._common_step(dev_batch, batch_idx)
    return loss, acc

  def validation_epoch_end(self, outputs):
    avg_loss = torch.mean(torch.stack([o[0] for o in outputs]))
    avg_acc = torch.mean(torch.stack([o[1] for o in outputs]))
    self.log_dict({'dev_loss': avg_loss, 'dev_acc': avg_acc},
      on_step=False, on_epoch=True, prog_bar=True, logger=True)

  def test_step(self, test_batch, batch_idx):
    loss, acc = self._common_step(test_batch, batch_idx)
    return loss, acc

  def test_epoch_end(self, outputs):
    avg_loss = torch.mean(torch.stack([o[0] for o in outputs]))
    avg_acc = torch.mean(torch.stack([o[1] for o in outputs]))
    # We don't log here because we might use multiple test dataloaders
    # and this causes an issue in logging
    results = {'loss': avg_loss.item(), 'acc': avg_acc.item()}
    # HACK: https://github.com/PyTorchLightning/pytorch-lightning/issues/1088
    self.test_results = results

  def predict_step(self, x):
    return self.forward(x)
