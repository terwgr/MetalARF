"""Strict resident nest force round trip and saved-halo face coupling."""
from __future__ import annotations
import numpy as np
from .array import MetalArray


def submit_round_trip(state, field, kind, halo):
    if not isinstance(field, MetalArray) or field.dtype != np.float32 or field.ndim != 3:
        raise TypeError('strict nest round trip requires a 3-D float32 Metal field')
    if kind not in (0,1,2,3,4): raise ValueError('unknown nest round-trip kind')
    nz,ny,nx=field.shape;mny,mnx=state.mup.shape
    params={'kind':kind,'has_msf':int(state.has_msf),'nz':nz,'ny':ny,'nx':nx,'mny':mny,'mnx':mnx}
    arrays={'field':field,'mub2d':state.mub2d,'mup':state.mup,'halo':halo,
            'c1h':state.c1h,'c2h':state.c2h,'c1f':state.c1f,'c2f':state.c2f,
            'msft':state.msft,'msfu':state.msfu,'msfv':state.msfv}
    return field.buffer.context.submit_kernel('strict_nest_roundtrip',arrays,params,
        exact_policy=True,grid_size=field.size,threads_per_group=128)


def submit_halo_face(state, field, kind, out, halo):
    if kind not in (0,1): raise ValueError('halo correction requires a wind face')
    if not isinstance(out,MetalArray) or tuple(out.shape)!=tuple(field.shape):
        raise ValueError('halo output must match the wind field')
    nz,ny,nx=field.shape;mny,mnx=state.mup.shape
    params={'kind':kind,'has_msf':int(state.has_msf),'nz':nz,'ny':ny,'nx':nx,'mny':mny,'mnx':mnx}
    arrays={'field':field,'out':out,'mub2d':state.mub2d,'mup':state.mup,
            'halo':halo,'c1h':state.c1h,'c2h':state.c2h,
            'msf':state.msfu if kind==0 else state.msfv}
    return out.buffer.context.submit_kernel('strict_nest_halo_face',arrays,params,
        exact_policy=True,grid_size=nz*(mny if kind==0 else mnx),threads_per_group=128)
