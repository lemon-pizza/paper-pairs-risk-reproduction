"""Rolling log-price regression and return summaries for the replay."""

import numpy as np
import pandas as pd

WINDOW = 504

def rolling_endpoint(x, y, window=WINDOW):
    """Complete-window intercept OLS; no imputation or future backfill."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.ndim == 1:
        x, y = x[:, None], y[:, None]
    ok = np.isfinite(x) & np.isfinite(y)
    xx, yy = np.where(ok, x, 0.), np.where(ok, y, 0.)

    def roll(a):
        cs = np.vstack([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0)])
        result = np.full_like(a, np.nan, dtype=float)
        if len(a) >= window:
            result[window - 1:] = cs[window:] - cs[:-window]
        return result

    count, sx, sy = roll(ok.astype(float)), roll(xx), roll(yy)
    variance = roll(xx * xx) - sx * sx / window
    covariance = roll(xx * yy) - sx * sy / window
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = covariance / variance
        alpha = (sy - beta * sx) / window
    beta[(count < window) | (variance < 1e-12)] = np.nan
    alpha[~np.isfinite(beta)] = np.nan
    spread = y - alpha - beta * x
    return beta, spread

def summarize_original_style(returns):
    """Preserve source semantics: Sharpe excludes NaN; cumprod fills NaN=0."""
    r = pd.Series(returns, dtype=float)
    return {
        'rows_including_duplicate_dates': len(r),
        'non_nan_rows': int(r.notna().sum()),
        'sharpe': float(np.sqrt(252) * r.mean() / r.std()),
        'sortino': float(np.sqrt(252) * r.mean() / r[r < 0].std()),
        'cumulative_return': float(((1 + r.fillna(0)).cumprod() - 1).iloc[-1]),
        'simple_return': float(r.fillna(0).cumsum().iloc[-1]),
    }
