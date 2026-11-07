#!/usr/bin/env python3
"""Build the unchanged native Noah text-table decoder for the Darwin wheel.

The existing woof.hex native/*.rs and libexec/* package rules include both
sources and artifacts. No Rust compiler is needed when running the wheel.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rustc', default='rustc')
    parser.add_argument('--clean', action='store_true', help='remove the staged decoder and manifest for a source-only build')
    args = parser.parse_args()
    if args.clean:
        for name in ('noah-table-decode', 'noah-table-decode.json'):
            (ROOT / 'woof/hex/libexec' / name).unlink(missing_ok=True)
        return
    if (platform.system(), platform.machine()) != ('Darwin', 'arm64'):
        raise RuntimeError('this helper builds the Apple Silicon Darwin artifact')
    source = ROOT / 'woof/hex/native'
    output = ROOT / 'woof/hex/libexec'
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='noah-decoder-build-') as directory:
        binary = Path(directory) / 'noah-table-decode'
        env = {**os.environ, 'ZERO_AR_DATE': '1'}
        subprocess.run([args.rustc, '--edition=2021', '--target=aarch64-apple-darwin',
                        '-O', '-C', 'debuginfo=0', '-C', 'strip=symbols',
                        '--remap-path-prefix', f'{ROOT}=.',
                        str(source / 'table_decode.rs'), '-o', str(binary)],
                       env=env, check=True)
        payload = binary.read_bytes()
        # Compiler diagnostics embedded in binaries must not reveal build roots.
        if any(value in payload for value in (str(ROOT).encode(), str(Path.home()).encode(),
                                              directory.encode())):
            raise ValueError('native Noah decoder contains a private build path')
        artifact = output / 'noah-table-decode'
        artifact.write_bytes(payload)
        artifact.chmod(0o755)
    manifest = dict(schema='mpas-noah-table-decoder/v1', target='aarch64-apple-darwin',
                    sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload),
                    sources={name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                             for name in ('table_decode.rs', 'table_schema.rs')})
    (output / 'noah-table-decode.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('staged native Noah decoder and source/artifact manifest')


if __name__ == '__main__':
    main()
