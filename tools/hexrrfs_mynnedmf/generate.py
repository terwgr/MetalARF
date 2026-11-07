#!/usr/bin/env python3
"""Generate the serial RRFS MYNN-EDMF Metal column from its pinned CUDA source."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STEPS = ('arrays.py', 'column.py', 'maths.py', 'entry.py')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--pin', choices=('c6d08709', 'c942418'), default='c6d08709')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output or root / 'woof/metal/shaders'
    env = dict(os.environ, WOOF_SOURCE_ROOT=str(root), WOOF_MYNN_PIN=args.pin)
    with tempfile.TemporaryDirectory(prefix='rrfs-mynn-generator-') as temporary:
        work = Path(temporary)
        for source in (HERE / 'helpers').iterdir():
            shutil.copyfile(source, work / source.name)
        for source in (HERE / 'templates').iterdir():
            shutil.copyfile(source, work / source.name)
        for step in STEPS:
            subprocess.run([sys.executable, str(work / step)], env=env, check=True,
                           stdout=subprocess.DEVNULL)
        generated = sorted((work / 'bundle').glob('hexrrfs_mynnedmf_*'))
        if len(generated) != (8 if args.pin == 'c6d08709' else 2):
            raise RuntimeError('unexpected RRFS MYNN output set')
        if args.check:
            changed = [file.name for file in generated
                       if not (output / file.name).is_file()
                       or file.read_bytes() != (output / file.name).read_bytes()]
            if changed:
                raise SystemExit('generated files differ: ' + ', '.join(changed))
        else:
            output.mkdir(parents=True, exist_ok=True)
            for file in generated:
                shutil.copyfile(file, output / file.name)
        print(f'{"Checked" if args.check else "Generated"} {len(generated)} RRFS MYNN files')


if __name__ == '__main__':
    main()
