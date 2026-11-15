"""Run the paper replay or its overlap and variance-constraint extension."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--paper', action='store_true', help='Reproduce MZ and BZ')
    mode.add_argument('--overlap', action='store_true', help='Compare M, CAP3 and RISK3 with Z signals')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    module = 'src.paper' if args.paper else 'src.paper_overlap'
    subprocess.run([sys.executable, '-m', module, '--workers', str(args.workers)],
                   env=env, cwd=Path(__file__).resolve().parent, check=True)


if __name__ == '__main__':
    main()
