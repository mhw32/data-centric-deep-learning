from typing import List
import torch
from .utils import fix_random_seed
from sklearn.cluster import KMeans

def random_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Randomly pick examples.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  fix_random_seed(42)
  
  indices = []
  # ================================
  # FILL ME OUT
  # Randomly pick a 1000 examples to label. This serves as a baseline.
  # Note that we fixed the random seed above. Please do not edit.
  pass
  # ================================
  return indices

def uncertainty_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the model is the least confident in its predictions.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  # ================================
  # FILL ME OUT
  # Sort indices by the predicted probabilities and choose the 1000 examples with 
  # the least confident predictions. Think carefully about what "least confident" means 
  # for a N-way classification problem.
  # Take the first 1000.
  pass
  # ================================
  return indices

def margin_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples where the difference between the top two predicted probabilities is the smallest.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  # ================================
  # FILL ME OUT
  # Sort indices by the different in predicted probabilities in the top two classes per example.
  # Take the first 1000.
  pass
  # ================================
  return indices

def entropy_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Pick examples with the highest entropy in the predicted probabilities.
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  indices = []
  # ================================
  # FILL ME OUT
  # Entropy is defined as -E_classes[log p(class | input)] aja the expected log probability
  # over all K classes. See https://en.wikipedia.org/wiki/Entropy_(information_theory).
  # Sort the indices by the entropy of the predicted probabilities from high to low.
  # Take the first 1000.
  pass
  # ================================
  return indices

def cluster_sampling(pred_probs: torch.Tensor, budget : int = 1000) -> List[int]:
  '''Cluster the inputs by predicted probabilities and pick the centroids of 
  :param pred_probs: list of predicted probabilities for the production set in order.
  :param budget: the number of examples you are allowed to pick for labeling.
  :return indices: A list of indices (into the `pred_probs`) for examples to label.
  '''
  fix_random_seed(42)

  indices = []
  # ================================
  # FILL ME OUT
  # Use k-means to cluster the predicted probabilities into 1000 clusters. Take a random
  # member for each cluster.
  # Use the `sklearn.cluster.KMeans` function to do clustering.
  # Note that we fixed the random seed above. Please do not edit.
  pass
  # ================================
  return indices
