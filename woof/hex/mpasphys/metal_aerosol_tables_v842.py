"""Authenticate and load native aerosol Thompson table words on Metal."""
from pathlib import Path
import hashlib
import numpy as np
from woof.ingest.rrtmg_coeffs import _read_fortran_record
from ..metal_dynamics_v841 import _arrays
from . import thompson_v842 as native, thompson_aero_v842 as aerosol
from .table_resources_v842 import certified_path, thompson_record_groups

# Closed until the independent official v1.1.5 Rust decoder establishes it.
AEROSOL_DECODED_SHA256 = "201f260387067ad0350824059fd963aecd91477f6ca9aa6a9dcd9563fed95b0c"


def _decode_raw(resources):
    directory = Path(resources['table_directory'])
    raw = {}
    digest = hashlib.sha256()
    for filename, names in thompson_record_groups(native):
        path = certified_path(dict(path=directory / filename, sha256=native.TABLE_FILE_SHA256[filename]))
        with path.open('rb') as stream:
            for record, name in enumerate(names, 1):
                payload = _read_fortran_record(stream, filename, record)
                count = int(np.prod(native.TABLE_SHAPES[name]))
                if len(payload) != count * 8:
                    raise ValueError(f'native Thompson record extent differs: {name}')
                words = np.frombuffer(payload, dtype='>u8').astype(np.uint64)
                raw[name] = words
                digest.update(name.encode() + b'\0' + words.tobytes())
            if stream.read(1):
                raise ValueError(f'native Thompson file has trailing records: {filename}')
    path = certified_path(dict(path=resources['auxiliary'], sha256=aerosol.AUXILIARY_SHA256))
    with path.open('rb') as stream:
        for name in native.TABLE_ARGUMENTS[27:]:
            width = 4 if name == 'tnccn_act' else 8
            count = int(np.prod(native.TABLE_SHAPES[name]))
            payload = stream.read(count * width)
            if len(payload) != count * width:
                raise ValueError(f'native aerosol auxiliary extent differs: {name}')
            words = np.frombuffer(payload, dtype='<u' + str(width)).copy()
            raw[name] = words
            digest.update(name.encode() + b'\0' + words.tobytes())
        if stream.read(1):
            raise ValueError('native aerosol auxiliary has trailing bytes')
    if set(raw) != set(native.TABLE_SHAPES):
        raise ValueError('native aerosol table set differs')
    return raw, digest.hexdigest()


def load_tables(resources, context):
    raw, digest = _decode_raw(resources)
    if digest != AEROSOL_DECODED_SHA256:
        raise ValueError('native aerosol decoded-output certificate differs')
    if sum(a.nbytes for a in raw.values()) > context.recommended_working_set_size:
        raise MemoryError('native Thompson tables exceed Metal recommended working set')
    xp = _arrays(context)
    tables = {name: xp.asarray(words.view(np.float32) if name == 'tnccn_act' else words)
              for name, words in raw.items()}
    tables['_file_sha256'] = dict(native.TABLE_FILE_SHA256)
    tables['_aerosol_auxiliary_sha256'] = aerosol.AUXILIARY_SHA256
    return tables
