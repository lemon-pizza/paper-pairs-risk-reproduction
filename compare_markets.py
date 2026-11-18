"""Build the two-market performance table from the completed daily backtests."""
import argparse
import json
from pathlib import Path

import pandas as pd

from src.risk3_adf_half import MODEL as HALF_MODEL, SUITE as HALF_SUITE
from src.selection import sha256
from src.trading import summarize_original_style


ROOT = Path(__file__).resolve().parent / 'results'


def market_rows(market, folder):
    baseline = json.loads((folder / 'paper/run.json').read_text())
    overlap = json.loads((folder / 'paper_overlap/run.json').read_text())
    suites = [('paper', 'strategy', ('MZ', 'BZ')), ('paper_overlap', 'model', ('CAP3', 'RISK3'))]
    runs = [('paper_overlap', overlap)]
    if (folder / HALF_SUITE).is_dir():
        extension = json.loads((folder / HALF_SUITE / 'run.json').read_text())
        for name in ('selections', 'monthly'):
            if extension[f'parent_{name}_sha256'] != sha256(folder / 'paper_overlap' / f'{name}.csv'):
                raise ValueError(f'{market}: the RISK3 extension uses different parent {name}')
        runs.append((HALF_SUITE, extension))
        suites.append((HALF_SUITE, 'model', (HALF_MODEL,)))
    for suite, run in runs:
        for key in ('price_sha256', 'first_date', 'last_date', 'trading_days', 'months', 'transaction_cost'):
            if baseline[key] != run[key]:
                raise ValueError(f'{market}: paper and {suite} runs differ in {key}')
    rows = []
    calendar = None
    for suite, key, names in suites:
        data = pd.read_csv(folder / suite / 'daily.csv', parse_dates=['date'])
        if suite == HALF_SUITE:
            data = data[data.execution.eq('signal_close')]
        for name in names:
            returns = data[data[key].eq(name)]
            dates = pd.DatetimeIndex(returns.date)
            if (len(dates) != baseline['trading_days'] or not dates.is_unique
                    or not dates.is_monotonic_increasing
                    or (calendar is not None and not calendar.equals(dates))):
                raise ValueError(f'{market}: incomplete daily returns for {name}')
            calendar = dates
            row = {'market': market, 'strategy': name, 'first_date': baseline['first_date'],
                   'last_date': baseline['last_date'], 'months': baseline['months'],
                   'trading_days': len(returns), 'one_way_bps': baseline['transaction_cost']['one_way_bps']}
            stats = {basis: summarize_original_style(returns[basis + '_return']) for basis in ('gross', 'net')}
            for metric in ('cumulative_return', 'annualized_return', 'sharpe'):
                for basis in ('gross', 'net'):
                    row[f'{basis}_{metric}'] = stats[basis][metric]
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sp500', type=Path, default=ROOT)
    parser.add_argument('--kospi', type=Path, default=ROOT / 'kospi')
    parser.add_argument('--output', type=Path, default=ROOT / 'market_comparison.csv')
    args = parser.parse_args()
    if (args.sp500 / HALF_SUITE).is_dir() != (args.kospi / HALF_SUITE).is_dir():
        parser.error('Run the RISK3 ADF top-half extension for both markets before combining the results')
    frame = pd.DataFrame(market_rows('S&P 500', args.sp500) + market_rows('KOSPI', args.kospi))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))


if __name__ == '__main__':
    main()
