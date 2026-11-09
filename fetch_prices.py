"""Install the checksum-pinned price input from the source repository or a local copy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('local_copy', type=Path, nargs='?')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    meta = json.loads((root / 'data/provenance.json').read_text())
    destination = root / meta['local_file']
    source = '/'.join((meta['source_repository'].replace('github.com', 'raw.githubusercontent.com'),
                       meta['source_commit'], meta['upstream_file']))
    if destination.is_file() and hashlib.sha256(destination.read_bytes()).hexdigest() == meta['sha256']:
        print('Price input already verified:', destination)
        return
    if args.local_copy:
        content = args.local_copy.read_bytes()
    else:
        with urlopen(source, timeout=45) as response:
            content = response.read()
    if hashlib.sha256(content).hexdigest() != meta['sha256']:
        raise SystemExit('Price input SHA-256 does not match the pinned source.')
    with NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    temporary.replace(destination)
    print('Verified price input:', destination)


if __name__ == '__main__':
    main()
