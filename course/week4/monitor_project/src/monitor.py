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
  # tr_heights, bin_edges = make histogram using tr_probs
  #   important for `density = True`.
  # te_heights, _ = make histogram using te_probs with the same 
  #   bin edges set to `bin_edges`. St `density = True`.
  # 
  # score = 0
  # loop though bins by index i
  #   bin_mid = compute middle of bin from two edges
  #   tr_area = bin_mid * tr_heights[i]
  #   te_area = bin_mid * te_heights[i]
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
  # 
  # Read the documentation for `np.histogram` carefully, in
  # particular what `bin_edges` represent.
  score = 0
  tr_heights, tr_bins = np.histogram(
    tr_probs.cpu().numpy(), density=True)
  te_heights, _ = np.histogram(
    te_probs.cpu().numpy(), bins=tr_bins, density=True)
  for i in range(len(tr_heights)):
    bin_mid = (tr_bins[i] + tr_bins[i+1]) / 2.
    tr_area = bin_mid * tr_heights[i]
    te_area = bin_mid * te_heights[i]
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

  def __init__(self, tr_vocab, tr_probs, tr_labels):
    self.tr_vocab = tr_vocab
    self.tr_probs = tr_probs
    self.tr_labels = tr_labels

  def calibrate(self, tr_probs, tr_labels, te_probs):
    tr_probs_cal = None
    te_probs_cal = None
    # ============================
    # FILL ME OUT
    # 
    # Calibrate probabilities with isotonic regression using 
    # the training probabilities and labels. 
    # 
    # Pseudocode:
    # --
    # use IsotonicRegression()
    # tr_probs_cal = fit calibration model
    # te_probs_cal = evaluate using fitted model
    # 
    # Type:
    # --
    # `tr_probs_cal`: torch.Tensor. Note that sklearn
    # returns a NumPy array. You will need to cast 
    # it to a torch.Tensor.
    # 
    # `te_probs_cal`: torch.Tensor
    cal_model = IsotonicRegression()
    tr_probs_cal = cal_model.fit_transform(
      tr_probs.cpu().numpy(), tr_labels.cpu().numpy())
    te_probs_cal = cal_model.transform(te_probs.cpu().numpy())
    tr_probs_cal = torch.from_numpy(tr_probs_cal).float()
    te_probs_cal = torch.from_numpy(te_probs_cal).float()
    # ============================
    return tr_probs_cal, te_probs_cal

  def monitor(self, te_vocab, te_probs):
    # tr_probs, te_probs = self.calibrate(self.tr_probs, self.tr_labels, te_probs)
    tr_probs = self.tr_probs
    # tr_labels = self.tr_labels
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
