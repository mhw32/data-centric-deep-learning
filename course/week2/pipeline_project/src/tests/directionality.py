"""
Directionality tests help you measure the 'robustness' of a trained 
model in terms of human expectations. These tests differ wildly by 
application. For handwritten digits application, we look at two 
different notions of robustness. 

1) A model should work just as well if the digit is rotated a little. 
2) A model should work just as well if a bit of noise is in the image.

In short, consistency is what we care about, not correctness -- which 
is the job of the integration and regression tests. Here, we care that
manual transformations of the image do not impact the model's predictions.
"""
import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from os.path import join
from PIL import Image
from skimage.util import random_noise
from torchvision import transforms

from torch.utils.data import DataLoader, TensorDataset
from src.tests.base import BaseTest


class GaussianNoise(object):
  """Add Gaussian Noise to an image."""
  def __init__(self, mean=0., std=1.):
    self.std = std
    self.mean = mean
    
  def __call__(self, tensor):
    return tensor + torch.randn(tensor.size()) * self.std + self.mean
  
  def __repr__(self):
    return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class MNISTDirectionalityTest(BaseTest):
  r"""A directionality test includes a set of tuples, with each 
  tuple containing two data points (one untransformed and one 
  transformed). The test checks if the model's prediction changes
  between the two tuples. 

  Arguments
  ---------
  num_aug (int, default=5): number of copies of the image to apply
    random augmentations to.
  """
  def __init__(self, num_aug = 5):
    super().__init__()
    self.num_aug = num_aug

    # we will reuse the same examples from the integration test
    base_paths = self.load_integration_examples()
    images_raw, images_transformed = self.create_directionality_tuples(base_paths)

    dataset = TensorDataset(images_raw, images_transformed)
    assert len(dataset) == 100, f"Unexpected dataset size: {len(dataset)}"
    self.dataset = dataset

  def load_integration_examples(self):
    r"""Returns paths for handwritten digits from the integration test. 
    We do not need the labels.

    Returns
    -------
    paths (list[str]): list of image paths belonging to integration
      examples.
    """
    test_dir = join(self.root, 'integration')
    paths = glob(join(test_dir, 'digits-processed', '*.png'))
    return paths

  def create_directionality_tuples(self, paths):
    r"""For every path combo, create `num_aug*2` tuples

      do num_aug times:
        pick random noise
        add (original image, image + random noise)
        pick random rotation
        add (original image, image + random rotation) 
    
    where the random noise and rotation are randomly sampled. Each
    time you run this test, you may get slightly different results.

    Returns
    -------
    images1 (torch.Tensor): original images
      shape: 10*num_aug*2 x 1 x 28 x 28
    images2 (torch.Tensor): directionally transformed images
      10*num_aug*2 x 1 x 28 x 28
    """
    standard_transform = transforms.ToTensor()
    noise_transform = transforms.Compose([
      transforms.ToTensor(),
      lambda x: torch.from_numpy(random_noise(x)).float(),
    ])
    rotate_transform = transforms.Compose([
      transforms.RandomRotation(45),  # max rotation is -45 -> 45 deg
      transforms.ToTensor(),
    ])

    images_raw, images_transformed = [], []

    # Use the transforms defined above to populate `images_raw`
    # and `images_transformed`.
    for path in paths:
      img = Image.open(path)
      orig = standard_transform(img)

      for _ in range(self.num_aug):
        noised = noise_transform(img)
        images_raw.append(orig)
        images_transformed.append(noised)

        rotated = rotate_transform(img)
        images_raw.append(orig)
        images_transformed.append(rotated)

    assert len(images_raw) == (len(paths) * self.num_aug * 2), \
      "Unexpected number of elements in `images_raw`."
    assert len(images_transformed) == (len(paths) * self.num_aug * 2), \
      "Unexpected number of elements in `images_raw`."

    images_raw = torch.stack(images_raw)
    images_transformed = torch.stack(images_transformed)

    return images_raw, images_transformed

  def get_dataloader(self, batch_size = 10):
    loader = DataLoader(self.dataset, batch_size=batch_size)
    return loader

  def test(self, trainer, system):
    loader = self.get_dataloader()
    pbar = tqdm(total = len(loader), leave = True, position = 0)

    metric = []
    for batch in loader:
      image_raw, image_transformed = batch

      logits_raw = system.predict_step(image_raw)
      logits_transformed = system.predict_step(image_transformed)
      preds_raw = torch.argmax(logits_raw, dim=1)
      preds_transformed = torch.argmax(logits_transformed, dim=1)

      batch_metric = 0  # store metric here
      # ================================
      # FILL ME OUT
      # 
      # Compute the fraction of times the transformed images maintains 
      # the same prediction. Store this in the `batch_metric` variable.
      # 
      # Make sure batch_metric is a floating point number, not a torch.Tensor.
      # You can extract a value from a torch.Tensor with `.item()`.
      # 
      # Our solution is one line of code.
      # 
      # Pseudocode:
      # --
      # batch_metric = ...
      # 
      # Type:
      # --
      # batch_metric: float (not torch.Tensor!)
      #   Metric computed on a minibatch
      # ================================
      metric.append(batch_metric)
      pbar.update()
    pbar.close()

    results = {'acc': float(np.mean(metric))}
    system.test_results = results
