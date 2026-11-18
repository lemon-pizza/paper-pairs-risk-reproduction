"""Shared inputs and ADF helpers for the paper replay."""

import hashlib
from pathlib import Path

import numpy as np
from scipy.special import ndtr
from statsmodels.tsa import adfvalues

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_PRICES = ROOT / 'data/prices.parquet.gz'

EXPECTED_SHA256 = '7605eab6525ddb5db66965d1559a56b48375f11a14cd1040eaa12318a545939c'

def adf_pvalues(t):
    t = np.asarray(t)
    small = np.polyval(adfvalues._tau_smallps['c'][0][::-1], t)
    large = np.polyval(adfvalues._tau_largeps['c'][0][::-1], t)
    out = ndtr(np.where(t <= adfvalues._tau_stars['c'][0], small, large))
    return np.where(t > adfvalues._tau_maxs['c'][0], 1.,
                    np.where(t < adfvalues._tau_mins['c'][0], 0., out))

def _design(e, lag):
    diff = np.diff(e, axis=1)
    y = diff[:, lag:].copy()
    cols = [e[:, lag:-1]]
    cols.extend(diff[:, lag-j:-j] for j in range(1, lag+1))
    x = np.stack(cols, axis=2)
    # Frisch-Waugh elimination of the intercept retains the exact regressions.
    y -= y.mean(axis=1, keepdims=True)
    x -= x.mean(axis=1, keepdims=True)
    return x, y

def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
