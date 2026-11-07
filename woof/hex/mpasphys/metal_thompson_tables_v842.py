"""Authenticate native Thompson table words and upload their exact bit patterns.

Host record transport replaces the CUDA path's Rust decoder. The file hashes,
array/file association and aggregate decoded-output certificate remain shared.
No initialization or scheme arithmetic is evaluated here.
"""
from pathlib import Path
import hashlib
import numpy as np
from woof.ingest.rrtmg_coeffs import _read_fortran_record
from ..metal_dynamics_v841 import _arrays
from . import thompson_v842 as native
from .table_resources_v842 import (certified_path, thompson_record_groups,
                                  THOMPSON_AUXILIARY_NAMES, THOMPSON_AUXILIARY_SHA256)


def load_raw_tables(resources):
    from .resources_build import DECODED_SHA256
    directory=Path(resources['table_directory']);raw={};digest=hashlib.sha256()
    for filename,names in thompson_record_groups(native):
        path=certified_path(dict(path=directory/filename,sha256=native.TABLE_FILE_SHA256[filename]))
        with path.open('rb') as stream:
            for record,name in enumerate(names,1):
                payload=_read_fortran_record(stream,filename,record)
                count=int(np.prod(native.TABLE_SHAPES[name]))
                if len(payload)!=count*8:
                    raise ValueError(f'native Thompson record extent differs: {name}')
                # Integer endian conversion preserves all IEEE payload bits.
                words=np.frombuffer(payload,dtype='>u8').astype(np.uint64)
                raw[name]=words;digest.update(name.encode()+b'\0'+words.tobytes())
            if stream.read(1):
                raise ValueError(f'native Thompson file has trailing records: {filename}')
    path=certified_path(dict(path=resources.get('auxiliary',directory/'auxiliary.bin'),sha256=THOMPSON_AUXILIARY_SHA256))
    with path.open('rb') as stream:
        for name in THOMPSON_AUXILIARY_NAMES:
            width=4 if name=='tnccn_act' else 8;count=int(np.prod(native.TABLE_SHAPES[name]))
            payload=stream.read(count*width)
            if len(payload)!=count*width:
                raise ValueError(f'native Thompson auxiliary extent differs: {name}')
            words=np.frombuffer(payload,dtype='<u'+str(width)).copy()
            raw[name]=words;digest.update(name.encode()+b'\0'+words.tobytes())
        if stream.read(1):
            raise ValueError('native Thompson auxiliary has trailing bytes')
    if set(raw)!=set(native.TABLE_SHAPES) or digest.hexdigest()!=DECODED_SHA256['mp_thompson']:
        raise ValueError('native Thompson decoded-output certificate differs')
    return raw


def load_tables(resources,context):
    """Return Fortran-linear uint64 table words plus the native float32 table."""
    raw=load_raw_tables(resources)
    if sum(a.nbytes for a in raw.values())>context.recommended_working_set_size:
        raise MemoryError('native Thompson tables exceed Metal recommended working set')
    xp=_arrays(context)
    tables={name:xp.asarray(words.view(np.float32) if name=='tnccn_act' else words)
            for name,words in raw.items()}
    tables['_file_sha256']=dict(native.TABLE_FILE_SHA256)
    return tables
