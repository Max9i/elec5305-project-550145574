"""Self-check for eer():  python python/test_eer.py"""
import numpy as np
from scipy.stats import norm

from evaluate import eer

t = np.r_[np.ones(50_000), np.zeros(50_000)]
assert eer(t, t) == 0     # perfect separation
assert eer(-t, t) == 100  # perfectly inverted scores
s = np.random.default_rng(0).normal(size=t.size) + 2 * t  # targets shifted by d' = 2
assert abs(eer(s, t) - 100 * norm.cdf(-1)) < 0.5, eer(s, t)  # theory: EER = Phi(-d'/2) = 15.87 %
print("eer ok")
