r"""A PyTorch Lightning system for training MNIST."""

import os
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from src.dataset import ProductReviewEmbeddings


class ReviewDataModule(pl.LightningDataModule):
  r"""Data module wrapper around review datasets."""

  def __init__(self, config, weights = None):
    super().__init__()

    # This should remind you a lot of the MNISTDataModule
    # but instead we load our custom dataset here.
    train_dataset = ProductReviewEmbeddings(
      lang=config.system.data.lang,
      split='train',
      weights = weights)
    dev_dataset = ProductReviewEmbeddings(
      lang=config.system.data.lang,
      split='dev')
    test_dataset = ProductReviewEmbeddings(
      lang=config.system.data.lang,
      split='test')

    self.train_dataset = train_dataset
    self.dev_dataset = dev_dataset
    self.test_dataset = test_dataset
    self.batch_size = config.system.optimizer.batch_size
    self.num_workers = config.system.optimizer.num_workers

  def train_dataloader(self):
    # Create a dataloader for train dataset. 
    # Notice we set `shuffle=True` for the training dataset.
    return DataLoader(self.train_dataset, batch_size = self.batch_size,
      shuffle = True, num_workers = self.num_workers)

  def val_dataloader(self):
    return DataLoader(self.dev_dataset, batch_size = self.batch_size,
      num_workers = self.num_workers)

  def test_dataloader(self):
    return DataLoader(self.test_dataset, batch_size = self.batch_size,
      num_workers = self.num_workers)


class SentimentClassifierSystem(pl.LightningModule):
  """A Pytorch Lightning system to train a model to classify sentiment of 
  product reviews. 

  Arguments
  ---------
  config (dotmap.DotMap): a configuration file with hyperparameters.
    See config.py for an example.
  """
  def __init__(self, config):
    super().__init__()
    self.save_hyperparameters()
    self.config = config

    # load model
    self.model = self.get_model()

    # We will overwrite this once we run `test()`
    self.test_results = {}

  def get_model(self):
    model = nn.Sequential(
      nn.Linear(768, self.config.system.model.width),
      nn.ReLU(),
      nn.Linear(self.config.system.model.width, 1)
    )
    return model
  
  def configure_optimizers(self):
    optimizer = optim.Adam(self.parameters(), lr=self.config.system.optimizer.lr)
    return optimizer

  def _common_step(self, batch, _):
    """
    Arguments
    ---------
    embs (torch.Tensor): embeddings of review text
      shape: batch_size x 768
    labels (torch.LongTensor): binary labels (0 or 1)
      shape: batch_size
    """
    embs, labels = batch['embedding'], batch['label']
    
    # forward pass using the model
    logits = self.model(embs)

    # https://pytorch.org/docs/stable/generated/torch.nn.functional.binary_cross_entropy_with_logits.html
    loss = F.binary_cross_entropy_with_logits(logits.squeeze(1), labels.float())

    with torch.no_grad():
      # Compute accuracy using the logits and labels
      preds = torch.round(torch.sigmoid(logits))
      num_correct = torch.sum(preds.squeeze(1) == labels)
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

  def predict_step(self, batch, _):
    logits = self.model(batch['embedding'])
    probs = torch.sigmoid(logits)
    return probs


class RobustSentimentSystem(SentimentClassifierSystem):
  """A Pytorch Lightning system to train a model to classify sentiment of 
  product reviews using DRO (assuming group knowledge).

  Arguments
  ---------
  config (dotmap.DotMap): a configuration file with hyperparameters.
    See config.py for an example.
  """

  def _common_step(self, batch, _):
    """
    Arguments
    ---------
    embs (torch.Tensor): embeddings of review text
      shape: batch_size x 768
    labels (torch.LongTensor): binary labels (0 or 1)
      shape: batch_size
    groups (torch.LongTensor): english or spanish (0 or 1)
      shape: batch_size
    """
    embs = batch['embedding']
    labels = batch['label']
    # `groups` is a tensor of 0s and 1s where 1 indicates the training
    # example is in the English group. 
    groups = batch['group']

    logits = self.model(embs)

    # compute loss per element (no reduction)
    loss = F.binary_cross_entropy_with_logits(
      logits.squeeze(1), labels.float(), reduction='none')

    # =================================
    # FILL ME OUT
    # 
    # Compute the DRO objective. The variable `loss` above 
    # is a torch.FloatTensor of the same length as the minibatch.
    # 
    # Write code to compute the average loss per group. Then 
    # compute the maximum over the group averages. Overwrite the 
    # `loss` variable with this value. This resulting loss is the 
    # one we will optimize using SGD.
    # 
    # Pseudocode:
    # --
    # loss0 = mean of terms in `loss` belonging to group 0
    # loss1 = mean of terms in `loss` belonging to group 1
    # loss = max(loss0, loss1)
    # 
    # Types:
    # --
    # loss0: torch.Tensor (length = # of group 0 elements in batch)
    # loss1: torch.Tensor (length = # of group 1 elements in batch)
    # loss: torch.Tensor (single element)
    # =================================

    with torch.no_grad():
      # Compute accuracy using the logits and labels
      preds = torch.round(torch.sigmoid(logits))
      num_correct = torch.sum(preds.squeeze(1) == labels)
      num_total = labels.size(0)
      accuracy = num_correct / float(num_total)

    return loss, accuracy