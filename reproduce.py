"""Run the paper replay, overlap experiment, or the ADF top-half extension of RISK3."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

from src.paper import add_input_arguments


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--paper', dest='module', action='store_const', const='src.paper',
                      help='Reproduce MZ and BZ')
    mode.add_argument('--overlap', dest='module', action='store_const', const='src.paper_overlap',
                      help='Compare M, CAP3 and RISK3 with Z signals')
    mode.add_argument('--risk3-adf-half', dest='module', action='store_const', const='src.risk3_adf_half',
                      help="Trade the ADF top half of RISK3 using the completed overlap run")
    parser.add_argument('--workers', type=int, default=4)
    add_input_arguments(parser)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    env = os.environ.copy()
    for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'MKL_NUM_THREADS'):
        env[name] = '1'
    command = [sys.executable, '-m', args.module, '--workers', str(args.workers),
               '--output-root', str(args.output_root.resolve())]
    if args.prices is not None:
        command.extend(['--prices', str(args.prices.resolve())])
    subprocess.run(command, env=env, cwd=Path(__file__).resolve().parent, check=True)


if __name__ == '__main__':
    main()
