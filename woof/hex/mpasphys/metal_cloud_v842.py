"""MPAS v8.4.2 cld_fraction slot and radiation carrier preparation."""
from __future__ import annotations
import math
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _i, _require
from .metal_tendencies_v842 import _launch


def cloud_fraction(*, prepared, mesh_density, xland, len_disp, diag_cldfrac, radshared, radiation_prepared=None, bl_cloud=None, itimestep=1):
    """Run cld_fraction on solve columns and publish radiation carriers.

    Profiles use [level,cell] RKIND arrays, pressure in Pa, temperature in K,
    mixing ratios in kg/kg. ``diag_cldfrac`` is the persistent Registry
    diagnostic; the returned cloud carriers are updated at each radiation alarm.
    Other ``radshared`` keys, including ozone state, are left intact.
    """
    qv=prepared['qv']
    nlev,ncells=qv.shape
    shape=(nlev,ncells)
    if not math.isfinite(len_disp) or len_disp <= 0:
        raise ValueError('MPAS cloudiness requires a positive resolved length')
    for n in ('qv','qc','qi','qs','t','pres'):
        _require(n,prepared[n],np.float32,shape)
    for n,value in (('mesh_density',mesh_density),('xland',xland)):
        _require(n,value,np.float32,(ncells,))
    _require('diag_cldfrac',diag_cldfrac,np.float32,shape)
    context=_context(qv)
    if bl_cloud is not None:
        for name in ('qc_bl', 'qi_bl', 'cldfrac_bl'):
            _require(name, bl_cloud[name], np.float32, shape)
            if _context(bl_cloud[name]) is not context:
                raise ValueError('MYNN cloud fields belong to another Metal context')
    arrays=_arrays(context)
    out={n:arrays.empty(shape,np.float32) for n in ('qvrad','qcrad','qirad','qsrad','cldfrac')}
    out.update({n:arrays.empty((ncells,),np.float32) for n in ('dx','xland')})
    _launch('hexv842_mpas_cloud_fraction_f32',nlev*ncells,context,
            dict(qv=qv,qc=prepared['qc'],qi=prepared['qi'],qs=prepared['qs'],
                 temp=prepared['t'],pressure=prepared['pres'],density=mesh_density,xland=xland,
                 qvrad=out['qvrad'],qcrad=out['qcrad'],qirad=out['qirad'],qsrad=out['qsrad'],
                 dx=out['dx'],land=out['xland'],cldfrac=out['cldfrac'],diag_cldfrac=diag_cldfrac),
            dict(nlev=_i(nlev),ncells=_i(ncells),len_disp=np.float32(len_disp)))
    if bl_cloud is not None:
        # The strict kernel is selected explicitly: Hex arithmetic is independent
        # of the process-wide WRF_EXACT switch in the generic WRF launcher.
        # The pointwise ABI names its axes for column-first radiation inputs.
        # Bind their extents in storage order for the native level-first views.
        context.submit_kernel('wrf_exact_rrtmgp_mynn_cloud_merge',
            dict(qc=out['qcrad'], qi=out['qirad'], cldfra=out['cldfrac'],
                 qc_bl=bl_cloud['qc_bl'], qi_bl=bl_cloud['qi_bl'],
                 cldfra_bl=bl_cloud['cldfrac_bl']),
            dict(ncell=nlev*ncells, ncol=nlev, nlay=ncells,
                 has_cldfra=1, replace_cldfra=int(itimestep != 1)),
            exact_policy=True, grid_size=nlev*ncells, threads_per_group=128)
        diag_cldfrac[...] = out['cldfrac']
    if radiation_prepared is not None:
        for name,value in out.items():
            _require(name, radiation_prepared[name], np.float32, value.shape)
            radiation_prepared[name][...] = value
    radshared.update(out)
    return out


_CLOUD_FIELDS = ('qvrad','qcrad','qirad','qsrad','cldfrac','dx','xland')


def cloud_restart_state(radshared):
    """Copy cloud carriers at a completed driver boundary; ozone is separately owned."""
    return {name:radshared[name].to_numpy() for name in _CLOUD_FIELDS}


def restore_cloud_state(radshared, state, context):
    """Restore the cloud carriers without modifying other radiation state."""
    if set(state) != set(_CLOUD_FIELDS):
        raise ValueError('MPAS cloud carrier checkpoint field set differs')
    shape=state['cldfrac'].shape
    if len(shape) != 2 or any(state[n].dtype != np.float32 or state[n].shape != shape
                              for n in _CLOUD_FIELDS[:5]):
        raise ValueError('MPAS cloud carrier checkpoint profile layout differs')
    if any(state[n].dtype != np.float32 or state[n].shape != (shape[1],)
           for n in _CLOUD_FIELDS[5:]):
        raise ValueError('MPAS cloud carrier checkpoint surface layout differs')
    arrays=_arrays(context)
    radshared.update({name:arrays.asarray(state[name]) for name in _CLOUD_FIELDS})


def cloudiness_slot(call):
    """Execute the v8.4.2 cld_fraction slot using native solve-sized pool views."""
    if call.scheme != 'cld_fraction':
        raise ValueError(f'MPAS Metal cloud scheme {call.scheme!r} is not implemented')
    from .radshared_v842 import icloud_bl_enabled
    pools = call.pools
    enabled = icloud_bl_enabled(getattr(call, 'options', {}))
    bl_cloud = {name:pools['diag_physics'][name] for name in ('qc_bl','qi_bl','cldfrac_bl')} if enabled else None
    return cloud_fraction(prepared=pools['prepared'], mesh_density=pools['mesh']['meshDensity'],
                          xland=pools['sfc_input']['xland'], len_disp=pools['config_len_disp'],
                          diag_cldfrac=pools['diag_physics']['cldfrac'], radshared=pools.setdefault('radshared', {}),
                          radiation_prepared=pools.get('radiation_prepared'), bl_cloud=bl_cloud,
                          itimestep=call.itimestep if enabled else 1)
