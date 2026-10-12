"""Content bindings for the packaged Metal runtime and its build inputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

BUNDLE_SCHEMA = 'woof-metal-bundle-v2'
BUNDLE_FILENAMES = ('libwoof_metal.dylib', 'woof_default.metallib', 'woof_exact.metallib')
STALE_BUNDLE_MESSAGE = 'stale Metal bundle: run tools/build_metal_backend.py'


def metal_source_provenance(repo_root: Path) -> dict:
    """Bind public source bytes without depending on private Git ancestry.

    Generated ABI headers are derived from the registry/generator and need
    not be present in an installed release. Native sources and build options
    are inputs too, even when the shader sources have not changed.
    """
    root = Path(repo_root).resolve()
    metal = root / 'woof/metal'
    inputs = [metal / 'kernel_signatures.json',
              root / 'tools/build_metal_backend.py',
              root / 'tools/generate_metal_kernel_signatures.py',
              root / 'tools/metal_abi.py']
    for name in ('shaders', 'signature_fragments', 'native'):
        folder = metal / name
        if not folder.is_dir():
            raise OSError(f'Missing Metal build inputs: {name}')
        inputs.extend(p for p in folder.rglob('*') if p.is_file())
    digest = hashlib.sha256()
    files = {}
    for path in sorted(set(inputs), key=lambda p: p.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        key, content = relative.encode('utf-8'), path.read_bytes()
        digest.update(len(key).to_bytes(4, 'little'))
        digest.update(key)
        digest.update(len(content).to_bytes(8, 'little'))
        digest.update(content)
        files[relative] = {'size': len(content), 'sha256': hashlib.sha256(content).hexdigest()}
    return {'binding': 'sha256-build-inputs', 'sha256': digest.hexdigest(), 'files': files}


def metal_source_sha256(repo_root: Path) -> str:
    """Hash shader, registry, native and generator/compiler build inputs."""
    return metal_source_provenance(repo_root)['sha256']


def verify_metal_bundle(repo_root: Path, *,
                        artifact_paths: Mapping[str, Path] | None = None) -> None:
    """Reject source drift or changed artifacts before any native code loads.

    Optional resolved paths let the runtime check the bytes it will actually
    load, including environment/path overrides, against the same manifest.
    """
    root = Path(repo_root).resolve()
    bundle = root / 'woof/libexec/metal'
    try:
        manifest = json.loads((bundle / 'METAL_BUNDLE.json').read_text(encoding='utf-8'))
        if manifest['schema'] != BUNDLE_SCHEMA:
            raise ValueError('rebuild required for native/artifact content bindings')
        current = metal_source_provenance(root)
        if (manifest['metal_inputs_sha256'] != current['sha256']
                or manifest['source_provenance'] != current):
            raise ValueError('build input fingerprint differs')
        records = manifest['files']
        if set(records) != set(BUNDLE_FILENAMES):
            raise ValueError('bundle artifact inventory differs')
        if artifact_paths is not None and set(artifact_paths) != set(BUNDLE_FILENAMES):
            raise ValueError('resolved artifact inventory differs')
        for name in BUNDLE_FILENAMES:
            path = Path(artifact_paths[name]) if artifact_paths is not None else bundle / name
            content = path.read_bytes()
            record = records[name]
            if (type(record['size']) is not int or record['size'] != len(content)
                    or record['sha256'] != hashlib.sha256(content).hexdigest()):
                raise ValueError(f'{name} size or digest differs')
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        raise RuntimeError(f'{STALE_BUNDLE_MESSAGE} ({error})') from None
