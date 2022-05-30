import torch
import numpy as np
from scipy.stats import ks_2samp
from sklearn.isotonic import IsotonicRegression


def get_ks_score(tr_probs, te_probs):
  score = None
  # ============================
  # FILL ME OUT
  # 
  # Compute the p-value from the Kolmogorov-Smirnov test.
  # You may use the imported `ks_2samp`.
  # 
  # Pseudocode:
  # --
  # convert tr_prob to numpy
  # convert te_prob to numpy
  # apply ks_2samp
  # 
  # Type:
  # --
  # tr_probs: torch.Tensor
  #   predicted probabilities from training test
  # te_probs: torch.Tensor
  #   predicted probabilities from test test
  # score: float - between 0 and 1
  _, score = ks_2samp(
    tr_probs.cpu().numpy(), te_probs.cpu().numpy())
  # ============================
  return score


def get_hist_score(tr_probs, te_probs, bins=10):
  score = None
  # ============================
  # FILL ME OUT
  # 
  # Compute histogram intersection score. 
  # 
  # Pseudocode:
  # --
  # tr_heights, tr_bins = make histogram using tr_probs
  #   important for normed = True
  # te_heights, te_bins = make histogram using te_probs
  #   important for normed = True
  # tr_bins and te_bins must be same size
  # 
  # score = 0
  # loop though bins by index i
  #   tr_area = tr_bins[i] * tr_heights[i]
  #   te_area = te_bins[i] * te_heights[i]
  #   intersect = min(tr_area, te_area)
  #   score = score + intersect
  # 
  # Type:
  # --
  # tr_probs: torch.Tensor
  #   predicted probabilities from training test
  # te_probs: torch.Tensor
  #   predicted probabilities from test test
  # score: float - between 0 and 1
  # 
  # Notes:
  # --
  # Remember to normalize the histogram so heights
  # sum to one. See `np.histogram`. Also remember to 
  # use the same bins for `tr_probs` and `te_probs`.
  score = 0
  tr_heights, tr_bins = np.histogram(
    tr_probs.cpu().numpy(), bins=bins, normed=True)
  te_heights, _ = np.histogram(
    te_probs.cpu().numpy(), bins=tr_bins, normed=True)
  for i in range(len(tr_bins)):
    tr_area = tr_bins[i] * tr_heights[i]
    te_area = tr_bins[i] * te_heights[i]
    intersect = min(tr_area, te_area)
    score += intersect
  # ============================
  return score


def get_vocab_outlier(tr_vocab, te_vocab):
  score = None
  # ============================
  # FILL ME OUT
  # 
  # Compute the percentage of the test vocabulary
  # that does not appear in the training vocabulary. A score
  # of 0 would mean all of the words in the test vocab
  # appear in the training vocab. A score of 1 would mean
  # none of the new words have been seen before. 
  # 
  # Pseudocode:
  # --
  # num_seen = ...
  # num_total = ...
  # score = 1 - (num_seen / num_total)
  # 
  # Type:
  # --
  # tr_vocab: dict[str, int]
  #   Map from word to count for training examples
  # te_vocab: dict[str, int]
  #   Map from word to count for test examples
  # score: float (between 0 and 1)
  num_seen = 0
  num_total = 0
  for word, _ in te_vocab.items():
    if word in tr_vocab:
      num_seen += 1
    num_total += 1
  score = 1. - num_seen / float(num_total)
  # ============================
  return score


class MonitoringSystem:

  def __init__(self, tr_vocab, tr_probs):
    self.tr_vocab = tr_vocab
    self.tr_probs = tr_probs

  def calibrate(self, probs, labels):
    cal_probs = None
    # ============================
    # FILL ME OUT
    # 
    # Calibrate probabilities with isotonic regression.
    # 
    # Pseudocode:
    # --
    # use IsotonicRegression()
    # cal_probs = ...
    # 
    # Type:
    # --
    # `cal_probs`: torch.Tensor. Note that sklearn
    # returns a NumPy array. You will need to cast 
    # it to a torch.Tensor.
    cal_model = IsotonicRegression()
    cal_probs = cal_model.fit_transform(probs.cpu().numpy(), labels.cpu().numpy())
    cal_probs = torch.from_numpy(cal_probs).float()
    # ============================
    return cal_probs

  def monitor(self, te_vocab, te_probs):
    tr_probs = self.calibrate(self.tr_probs)
    te_probs = self.calibrate(te_probs)
    # ============================
    # FILL ME OUT
    # 
    # Compute metrics. 
    # 
    # Pseudocode:
    # --
    # ks_score = ...
    # hist_score = ...
    # outlier_score = ...
    # 
    # Type:
    # --
    # ks_score: float
    # hist_score: float
    # outlier_score: float
    ks_score = get_ks_score(tr_probs, te_probs)
    hist_score = get_hist_score(tr_probs, te_probs)
    outlier_score = get_vocab_outlier(self.tr_vocab, te_vocab)
    # ============================
    metrics = {
      'ks_score': ks_score,
      'hist_score': hist_score,
      'outlier_score': outlier_score,
    }
    return metrics
