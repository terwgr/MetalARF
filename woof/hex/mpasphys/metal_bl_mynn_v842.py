"""Native MPAS MYNN PBL components; the complete slot is not yet registered."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _context, _require
from .metal_tendencies_v842 import _launch


def _apply(kind, inputs, outputs, *, flags=None, delt=None, exact_inverse=False):
    first=next(iter(inputs.values()));shape=first.shape;context=_context(first)
    arrays={**inputs,**outputs}
    for name,value in arrays.items():
        _require(name,value,np.float32,shape)
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    count=int(first.size)
    if not count:return
    params={'count':np.int32(count)}
    if flags is not None:
        if len(flags)!=3:raise ValueError('MYNN moisture flags must contain qc, qi, qs')
        params.update({name:np.int32(value) for name,value in zip(('f_qc','f_qi','f_qs'),flags)})
    if delt is not None:params['delt']=np.float32(delt)
    suffix='_rp1' if exact_inverse else ''
    _launch(f'hexv842_mpas_bl_mynn_{kind}{suffix}_f32',count,context,arrays,params)


def pre(mixing, specific, flags, *, exact_inverse=False):
    """MPAS mixing ratios to specific contents; absent snow is left intact."""
    _apply('pre',{n:mixing[n] for n in ('qv','qc','qi','qs')},
           {n:specific[n] for n in ('sqv','sqc','sqi','sqs')},flags=flags,exact_inverse=exact_inverse)


def post(mixing, tendencies, flags, delt, *, exact_inverse=False):
    """MPAS specific tendencies to mixing-ratio tendencies in source order."""
    _apply('post',{n:mixing[n] for n in ('qv','qc','qi','qs')},
           {n:tendencies[n] for n in ('dqv','dqc','dqi','dqs')},flags=flags,delt=delt,exact_inverse=exact_inverse)


def radiation(lw, sw, result):
    """Combine longwave and shortwave heating in the native operand order."""
    _apply('radiation',{'lw':lw,'sw':sw},{'result':result})


def grid_spacing(mesh_density, length, result):
    """Native dx for positive normal density; exceptional pow fallback is closed."""
    from ..metal_dynamics_v841 import _arrays
    if mesh_density.ndim!=1:raise ValueError('MYNN mesh density must be one-dimensional')
    n=int(mesh_density.size);context=_context(mesh_density)
    _require('mesh_density',mesh_density,np.float32,(n,));_require('dx',result,np.float32,(n,))
    if _context(result) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    if not n:return
    invalid=_arrays(context).empty((n,),np.int32)
    _launch('hexv842_mpas_bl_mynn_dx_validate_f32',n,context,{'mesh_density':mesh_density,'invalid':invalid},{'ncol':np.int32(n)})
    if np.any(invalid.to_numpy()):raise ValueError('MYNN grid spacing requires positive normal mesh density; native exceptional pow fallback is not translated')
    _launch('hexv842_mpas_bl_mynn_dx_f32',n,context,{'mesh_density':mesh_density,'dx':result},{'ncol':np.int32(n),'len_disp':np.float32(length)})


def tridiag(variant, a, b, c, d, x=None):
    """Run one native MYNN tridiagonal solve on independent column rows.

    The caller supplies the source-valid systems and distinct array storage;
    variants 1 and 3 modify their native in/out arguments in place.
    """
    from ..metal_dynamics_v841 import _arrays
    if variant not in (1,2,3):raise ValueError('MYNN tridiagonal variant must be 1, 2 or 3')
    if a.ndim!=2:raise ValueError('MYNN tridiagonal arrays require (column,level)')
    nc,nl=map(int,a.shape);context=_context(a)
    if nc<1 or nl<1:raise ValueError('MYNN tridiagonal solve requires nonempty columns and levels')
    arrays={'a':a,'b':b,'c':c,'d':d}
    if variant!=1:
        if x is None:raise ValueError('MYNN tridiagonal variants 2 and 3 require x')
        arrays['x']=x
    for name,value in arrays.items():
        _require(name,value,np.float32,(nc,nl))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    xp=_arrays(context)
    if variant==1:arrays['q']=xp.empty((nc,nl),np.float32)
    if variant==2:
        arrays['cp']=xp.empty((nc,nl),np.float32);arrays['dp']=xp.empty((nc,nl),np.float32)
    _launch(f'hexv842_mpas_bl_mynn_tridiag{variant}_f32',nc,context,arrays,{'nc':np.int32(nc),'nl':np.int32(nl)})


def level2(fields, *, gravity, ep1):
    """Native level-2 closure from MYNN's initialized physical constants.

    Each field is column-major (column,level); level zero of each output is
    retained, matching the source's loop from kts+1 through kte.
    """
    names=('dz','u','v','thl','thetav','qw','ql','vt','vq','dtl','dqw','dtv','gm','gh','sm','sh')
    arrays={name:fields[name] for name in names};first=arrays['dz']
    if first.ndim!=2:raise ValueError('MYNN level-2 fields require (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<1:raise ValueError('MYNN level-2 closure needs columns and levels')
    for name,value in arrays.items():
        _require(name,value,np.float32,(nc,nl))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    _launch('hexv842_mpas_bl_mynn_level2_f32',nc,context,arrays,
            {'nc':np.int32(nc),'nl':np.int32(nl),'gravity':np.float32(gravity),'ep1':np.float32(ep1)})


def predict(fields, *, closure, delt, ust, flt, flq, pmz, phh, karman,
            edmf_tke, tkebudget):
    """Run the native MMM turbulence-state predictor, including its solves.

    Fields use independent contiguous (column,level) rows. The caller supplies
    source-valid positive layer depths, density and nonzero surface scales.
    """
    from ..metal_dynamics_v841 import _arrays
    if tkebudget:raise ValueError('MYNN TKE budget reads an uninitialized top-level bp in the native source')
    names=('dz','el','dfq','rho','pdk','pdt','pdq','pdc','s_aw','s_awqke',
           'qke','tsq','qsq','cov','qwt1d','qdiss1d')
    arrays={name:fields[name] for name in names};first=arrays['dz']
    if first.ndim!=2:raise ValueError('MYNN predictor fields require (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<2:raise ValueError('MYNN predictor needs columns and at least two levels')
    for name,value in arrays.items():
        _require(name,value,np.float32,(nc,nl))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    if 4*nc*18*(nl+1)>context.recommended_working_set_size:raise MemoryError('MYNN predictor scratch exceeds Metal working set')
    arrays['scratch']=_arrays(context).empty((nc,18,nl+1),np.float32)
    params={n:np.float32(v) for n,v in dict(closure=closure,delt=delt,ust=ust,flt=flt,flq=flq,pmz=pmz,phh=phh,karman=karman).items()}
    params.update(nc=np.int32(nc),nl=np.int32(nl),bl_mynn_edmf_tke=np.int32(edmf_tke),bl_mynn_tkebudget=np.int32(tkebudget))
    _launch('hexv842_mpas_bl_mynn_predict_f32',nc,context,arrays,params)


def stability(zet, dx, pblh, outputs):
    """Native MYNN surface and scale-aware factors on the defined domain.

    Inputs require finite zet and positive normal dx and boundary-layer height.
    Validation status is read by the host before any output is published.
    """
    from ..metal_dynamics_v841 import _arrays
    if zet.ndim!=1:raise ValueError('MYNN stability inputs must be one-dimensional')
    n=int(zet.size);context=_context(zet)
    arrays={'zet':zet,'dx':dx,'pblh':pblh,**{k:outputs[k] for k in ('phim','phih','psig_bl','psig_shcu')}}
    for name,value in arrays.items():
        _require(name,value,np.float32,(n,))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    if not n:return
    invalid=_arrays(context).empty((n,),np.int32)
    _launch('hexv842_mpas_bl_mynn_stability_validate_f32',n,context,{k:arrays[k] for k in ('zet','dx','pblh')}|{'invalid':invalid},{'count':np.int32(n)})
    if np.any(invalid.to_numpy()):raise ValueError('MYNN stability needs finite zet and positive normal dx and pblh')
    _launch('hexv842_mpas_bl_mynn_stability_f32',n,context,arrays,{'count':np.int32(n)})


def tanh_values(values, result):
    """Evaluate the pinned MYNN CORE-MATH tanhf primitive on float32 words."""
    if values.ndim!=1:raise ValueError('MYNN tanh values must be one-dimensional')
    count=int(values.size);context=_context(values)
    _require('values',values,np.float32,(count,));_require('result',result,np.float32,(count,))
    if _context(result) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    if count:_launch('hexv842_mpas_bl_mynn_tanh_f32',count,context,{'input':values,'output':result},{'count':np.int32(count)})


def pbl_height(thetav, qke, zw, dz, landsea, zi, kzi):
    """Native MYNN boundary-layer height for finite, source-bounded columns.

    The scan requires a height above 200 m before its allocation ends. Device
    validation is read by the host before outputs are written.
    """
    from ..metal_dynamics_v841 import _arrays
    if thetav.ndim!=2:raise ValueError('MYNN PBL height requires (column,level)')
    nc,nl=map(int,thetav.shape);context=_context(thetav)
    if nc<1 or nl<3:raise ValueError('MYNN PBL height needs columns and at least three levels')
    arrays={'thetav1d':thetav,'qke1d':qke,'zw1d':zw,'dz1d':dz,'landsea':landsea,'zi':zi,'kzi':kzi}
    for name,value in arrays.items():
        shape=(nc,nl) if name in ('thetav1d','qke1d','zw1d','dz1d') else (nc,)
        _require(name,value,np.int32 if name=='kzi' else np.float32,shape)
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    invalid=_arrays(context).empty((nc,),np.int32);params={'nc':np.int32(nc),'nl':np.int32(nl)}
    _launch('hexv842_mpas_bl_mynn_pblh_validate_f32',nc,context,{k:arrays[k] for k in ('thetav1d','qke1d','zw1d','dz1d','landsea')}|{'invalid':invalid},params)
    if np.any(invalid.to_numpy()):raise ValueError('MYNN PBL height needs finite columns with a height above 200 m')
    _launch('hexv842_mpas_bl_mynn_pblh_f32',nc,context,arrays,params)


def mixing_length2(fields, *, gravity, karman):
    """Native MYNN mixing-length selector 2, with source-safe height scan.

    Arrays are column-major; zw has one additional interface. The host reads
    validation status before publication. Other selectors are not implemented.
    """
    from ..metal_dynamics_v841 import _arrays
    names=('dz','u1','v1','qke','dtv','cldfra_bl1d','edmf_w1','edmf_a1','zw','zi','rmo','fltv','psig_bl','qkw','el')
    arrays={name:fields[name] for name in names};first=arrays['dz']
    if first.ndim!=2:raise ValueError('MYNN mixing length requires (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<2:raise ValueError('MYNN mixing length needs columns and at least two levels')
    for name,value in arrays.items():
        shape=(nc,nl+1) if name=='zw' else (nc,) if name in ('zi','rmo','fltv','psig_bl') else (nc,nl)
        _require(name,value,np.float32,shape)
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    for name,value in (('gravity',gravity),('karman',karman)):
        bits=np.asarray(np.float32(value)).view(np.uint32).item()
        if not 0x00800000<=bits<0x7f800000:raise ValueError(f'MYNN {name} must be positive normal')
    if 4*nc*nl>context.recommended_working_set_size:raise MemoryError('MYNN mixing-length scratch exceeds Metal working set')
    xp=_arrays(context);params={'nc':np.int32(nc),'nl':np.int32(nl),'gravity':np.float32(gravity),'karman':np.float32(karman)}
    invalid=xp.empty((nc,),np.int32)
    _launch('hexv842_mpas_bl_mynn_length2_validate_f32',nc,context,{k:v for k,v in arrays.items() if k not in ('qkw','el')}|{'invalid':invalid},params)
    if np.any(invalid.to_numpy()):raise ValueError('MYNN mixing length needs finite fields, positive depths and a source-safe height scan')
    arrays['qtke']=xp.empty((nc,nl),np.float32)
    _launch('hexv842_mpas_bl_mynn_length2_f32',nc,context,arrays,params)


def initialize(fields, *, gravity, karman, ep1, initialize_qke):
    """Run native MYNN cold-state initialization for mixing-length selector 2.

    Column-major inputs require finite source-valid fields, positive layer
    depths and surface friction velocity, and a terminating height scan.
    The caller supplies source-valid remaining inputs, including positive
    friction velocity. Height-scan validation is read on the host before
    publication; it does not validate every input.
    """
    from ..metal_dynamics_v841 import _arrays
    profiles=('dz','u','v','thl','qw','theta','thetav','cldfra_bl1d','edmf_w1','edmf_a1','sh','sm','el','qke','tsq','qsq','cov')
    columns=('zi','rmo','ust','psig_bl')
    arrays={name:fields[name] for name in profiles+('zw',)+columns};first=arrays['dz']
    if first.ndim!=2:raise ValueError('MYNN initialization requires (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<2:raise ValueError('MYNN initialization requires columns and at least two levels')
    for name,value in arrays.items():
        shape=(nc,nl+1) if name=='zw' else (nc,) if name in columns else (nc,nl)
        _require(name,value,np.float32,shape)
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    for name,value in (('gravity',gravity),('karman',karman),('ep1',ep1)):
        bits=np.asarray(np.float32(value)).view(np.uint32).item()
        if not 0x00800000<=bits<0x7f800000:raise ValueError(f'MYNN {name} must be positive normal')
    if 4*nc*(15*nl+2)>context.recommended_working_set_size:raise MemoryError('MYNN initialization scratch exceeds Metal working set')
    xp=_arrays(context);params={'nc':np.int32(nc),'nl':np.int32(nl),'gravity':np.float32(gravity),'karman':np.float32(karman)}
    zero=xp.zeros((nc,nl),np.float32);zeros=xp.zeros((nc,),np.float32);invalid=xp.empty((nc,),np.int32)
    check={'dz':arrays['dz'],'u1':arrays['u'],'v1':arrays['v'],'qke':arrays['qke'],'dtv':zero,'cldfra_bl1d':arrays['cldfra_bl1d'],'edmf_w1':arrays['edmf_w1'],'edmf_a1':arrays['edmf_a1'],'zw':arrays['zw'],'zi':arrays['zi'],'rmo':arrays['rmo'],'fltv':zeros,'psig_bl':arrays['psig_bl'],'invalid':invalid}
    _launch('hexv842_mpas_bl_mynn_length2_validate_f32',nc,context,check,params)
    if np.any(invalid.to_numpy()):raise ValueError('MYNN initialization requires finite source-safe mixing-length inputs')
    arrays['scratch']=xp.empty((nc,14,nl),np.float32)
    params.update(ep1=np.float32(ep1),initialize_qke=np.int32(initialize_qke))
    _launch('hexv842_mpas_bl_mynn_initialize_f32',nc,context,arrays,params)


def turbulence25(fields, *, gravity, karman, ep1):
    """Native MYNN closure 2.5 with selector 2 and no TKE budget or SPP.

    The caller supplies source-valid inputs; the height scan is checked before
    publication. Status readback for that check takes place on the host.
    """
    from ..metal_dynamics_v841 import _arrays
    inputs=('dz','u','v','thl','thetav','qw','ql','vt','vq','qke','tsq','qsq','cov','theta','cldfra_bl1d','edmf_w1','edmf_a1','tkeprodtd')
    outputs=('sh','sm','el','dfm','dfh','dfq','tcd','qcd','pdk','pdt','pdq','pdc','qshear1d','qbuoy1d')
    columns=('zi','rmo','fltv','psig_bl')
    arrays={n:fields[n] for n in inputs+outputs+('zw',)+columns};first=arrays['dz']
    if first.ndim!=2:raise ValueError('MYNN turbulence requires (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<2:raise ValueError('MYNN turbulence needs columns and at least two levels')
    for name,value in arrays.items():
        shape=(nc,nl+1) if name=='zw' else (nc,) if name in columns else (nc,nl)
        _require(name,value,np.float32,shape)
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    for name,value in (('gravity',gravity),('karman',karman),('ep1',ep1)):
        bits=np.asarray(np.float32(value)).view(np.uint32).item()
        if not 0x00800000<=bits<0x7f800000:raise ValueError(f'MYNN {name} must be positive normal')
    if 4*nc*(8*nl+1)>context.recommended_working_set_size:raise MemoryError('MYNN turbulence scratch exceeds Metal working set')
    xp=_arrays(context);params={'nc':np.int32(nc),'nl':np.int32(nl),'gravity':np.float32(gravity),'karman':np.float32(karman)}
    zero=xp.zeros((nc,nl),np.float32);invalid=xp.empty((nc,),np.int32)
    check={'dz':arrays['dz'],'u1':arrays['u'],'v1':arrays['v'],'qke':arrays['qke'],'dtv':zero,'cldfra_bl1d':arrays['cldfra_bl1d'],'edmf_w1':arrays['edmf_w1'],'edmf_a1':arrays['edmf_a1'],'zw':arrays['zw'],'zi':arrays['zi'],'rmo':arrays['rmo'],'fltv':arrays['fltv'],'psig_bl':arrays['psig_bl'],'invalid':invalid}
    _launch('hexv842_mpas_bl_mynn_length2_validate_f32',nc,context,check,params)
    if np.any(invalid.to_numpy()):raise ValueError('MYNN turbulence requires finite source-safe mixing-length inputs')
    arrays['scratch']=xp.empty((nc,7,nl),np.float32);params['ep1']=np.float32(ep1)
    _launch('hexv842_mpas_bl_mynn_turbulence25_f32',nc,context,arrays,params)


def initialize_length1(fields, *, gravity, karman, ep1, initialize_qke):
    """Initialize native MYNN using mixing-length selector 1.

    The caller supplies source-valid physical inputs. Validate the depth and
    height scan before changing any output fields.
    """
    from ..metal_dynamics_v841 import _arrays

    profiles = ('dz', 'u', 'v', 'thl', 'qw', 'theta', 'thetav',
                'cldfra_bl1d', 'edmf_w1', 'edmf_a1', 'sh', 'sm', 'el',
                'qke', 'tsq', 'qsq', 'cov')
    columns = ('zi', 'rmo', 'ust', 'psig_bl')
    arrays = {name: fields[name] for name in profiles + ('zw',) + columns}
    first = arrays['dz']
    if first.ndim != 2:
        raise ValueError('MYNN initialization requires (column,level)')
    nc, nl = map(int, first.shape)
    context = _context(first)
    if nc < 1 or nl < 2:
        raise ValueError('MYNN initialization requires columns and at least two levels')
    for name, value in arrays.items():
        shape = (nc, nl + 1) if name == 'zw' else (nc,) if name in columns else (nc, nl)
        _require(name, value, np.float32, shape)
        if _context(value) is not context:
            raise ValueError('MYNN arrays belong to different Metal contexts')
    for name, value in (('gravity', gravity), ('karman', karman), ('ep1', ep1)):
        bits = np.asarray(np.float32(value)).view(np.uint32).item()
        if not 0x00800000 <= bits < 0x7f800000:
            raise ValueError(f'MYNN {name} must be positive normal')
    working_bytes = 4 * nc * (20 * nl + 2)
    scratch_bytes = 4 * nc * 19 * nl
    if (working_bytes > context.recommended_working_set_size
            or max(scratch_bytes, 4 * nc * nl, 4 * nc) > context.max_buffer_length):
        raise MemoryError('MYNN initialization scratch exceeds Metal limits')
    xp = _arrays(context)
    params = {'nc': np.int32(nc), 'nl': np.int32(nl),
              'gravity': np.float32(gravity), 'karman': np.float32(karman)}
    zero = xp.zeros((nc, nl), np.float32)
    zeros = xp.zeros((nc,), np.float32)
    invalid = xp.empty((nc,), np.int32)
    check = {'dz': arrays['dz'], 'u1': arrays['u'], 'v1': arrays['v'],
             'qke': arrays['qke'], 'dtv': zero, 'theta': arrays['theta'],
             'edmf_w1': arrays['edmf_w1'], 'edmf_a1': arrays['edmf_a1'],
             'zw': arrays['zw'], 'zi': arrays['zi'], 'rmo': arrays['rmo'],
             'fltv': zeros, 'psig_bl': arrays['psig_bl'], 'invalid': invalid}
    _launch('hexv842_mpas_bl_mynn_length1_validate_f32', nc, context, check, params)
    if np.any(invalid.to_numpy()):
        raise ValueError('MYNN initialization requires finite source-safe mixing-length inputs')
    arrays['scratch'] = xp.empty((nc, 19, nl), np.float32)
    params.update(ep1=np.float32(ep1), initialize_qke=np.int32(initialize_qke))
    _launch('hexv842_mpas_bl_mynn_initialize_length1_f32', nc, context, arrays, params)


def turbulence26_length1(fields, *, gravity, karman, ep1):
    """Run native MYNN turbulence with closure 2.6 and mixing length 1.

    The caller supplies source-valid physical inputs. Validate the depth and
    height scan before changing any output fields.
    """
    from ..metal_dynamics_v841 import _arrays

    inputs = ('dz', 'u', 'v', 'thl', 'thetav', 'qw', 'ql', 'vt', 'vq', 'qke',
              'tsq', 'qsq', 'cov', 'theta', 'cldfra_bl1d', 'edmf_w1',
              'edmf_a1', 'tkeprodtd')
    outputs = ('sh', 'sm', 'el', 'dfm', 'dfh', 'dfq', 'tcd', 'qcd', 'pdk',
               'pdt', 'pdq', 'pdc', 'qshear1d', 'qbuoy1d')
    columns = ('zi', 'rmo', 'fltv', 'psig_bl')
    arrays = {name: fields[name] for name in inputs + outputs + ('zw',) + columns}
    first = arrays['dz']
    if first.ndim != 2:
        raise ValueError('MYNN turbulence requires (column,level)')
    nc, nl = map(int, first.shape)
    context = _context(first)
    if nc < 1 or nl < 2:
        raise ValueError('MYNN turbulence needs columns and at least two levels')
    for name, value in arrays.items():
        shape = (nc, nl + 1) if name == 'zw' else (nc,) if name in columns else (nc, nl)
        _require(name, value, np.float32, shape)
        if _context(value) is not context:
            raise ValueError('MYNN arrays belong to different Metal contexts')
    for name, value in (('gravity', gravity), ('karman', karman), ('ep1', ep1)):
        bits = np.asarray(np.float32(value)).view(np.uint32).item()
        if not 0x00800000 <= bits < 0x7f800000:
            raise ValueError(f'MYNN {name} must be positive normal')
    working_bytes = 4 * nc * (13 * nl + 1)
    scratch_bytes = 4 * nc * 12 * nl
    if (working_bytes > context.recommended_working_set_size
            or max(scratch_bytes, 4 * nc * nl, 4 * nc) > context.max_buffer_length):
        raise MemoryError('MYNN turbulence scratch exceeds Metal limits')
    xp = _arrays(context)
    params = {'nc': np.int32(nc), 'nl': np.int32(nl),
              'gravity': np.float32(gravity), 'karman': np.float32(karman)}
    zero = xp.zeros((nc, nl), np.float32)
    invalid = xp.empty((nc,), np.int32)
    check = {'dz': arrays['dz'], 'u1': arrays['u'], 'v1': arrays['v'],
             'qke': arrays['qke'], 'dtv': zero, 'theta': arrays['theta'],
             'edmf_w1': arrays['edmf_w1'], 'edmf_a1': arrays['edmf_a1'],
             'zw': arrays['zw'], 'zi': arrays['zi'], 'rmo': arrays['rmo'],
             'fltv': arrays['fltv'], 'psig_bl': arrays['psig_bl'], 'invalid': invalid}
    _launch('hexv842_mpas_bl_mynn_length1_validate_f32', nc, context, check, params)
    if np.any(invalid.to_numpy()):
        raise ValueError('MYNN turbulence requires finite source-safe mixing-length inputs')
    arrays['scratch'] = xp.empty((nc, 12, nl), np.float32)
    params['ep1'] = np.float32(ep1)
    _launch('hexv842_mpas_bl_mynn_turbulence26_length1_f32', nc, context, arrays, params)


def log10_values(values, result):
    """Evaluate the pinned MYNN CORE-MATH log10f primitive on float32 words."""
    if values.ndim!=1:raise ValueError('MYNN log10 values must be one-dimensional')
    count=int(values.size);context=_context(values)
    _require('values',values,np.float32,(count,));_require('result',result,np.float32,(count,))
    if _context(result) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    if count:_launch('hexv842_mpas_bl_mynn_log10_f32',count,context,{'input':values,'output':result},{'count':np.int32(count)})


def condensation2(fields, *, constants):
    """Native cloud-PDF selector 2 condensation with stochastic option off.

    Inputs are source-valid column-major profiles with positive-normal
    pressure divided by 101325 and valid thermodynamic denominators. This partial adapter does not
    register the full PBL slot or validate every physical input.
    """
    from ..metal_dynamics_v841 import _arrays
    inputs=('dz','zw','thl','qw','qv','qc','qi','qs','p','exner','tsq','qsq','cov','sh','el','th','rstoch_col')
    outputs=('qc_bl1d','qi_bl1d','cldfra_bl1d','vt','vq','sgm')
    columns=('xland','pblh1','hfx1','rmo')
    scalars=('cp','cpv','cice','cliq','ep_2','r_d','r_v','xls','xlv','t0c','xlvcp','ep1')
    arrays={n:fields[n] for n in inputs+outputs+columns};first=arrays['dz']
    if first.ndim!=2:raise ValueError('MYNN condensation requires (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<3:raise ValueError('MYNN condensation needs columns and at least three levels')
    for name,value in arrays.items():
        _require(name,value,np.float32,(nc,) if name in columns else (nc,nl))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    if 4*nc*7*nl>context.recommended_working_set_size:raise MemoryError('MYNN condensation scratch exceeds Metal working set')
    arrays['scratch']=_arrays(context).empty((nc,7,nl),np.float32)
    params={n:np.float32(constants[n]) for n in scalars};params.update(nc=np.int32(nc),nl=np.int32(nl))
    _launch('hexv842_mpas_bl_mynn_condensation2_f32',nc,context,arrays,params)


def moisture_check(fields, *, delt, xlvcp, xlscp):
    """Native MYNN column moisture correction, including downward carry.

    The caller supplies source-valid positive layer pressure increments,
    nonzero Exner values and timestep, with distinct array storage.
    """
    names=('dp','exner','qv','qc','qi','qs','th','dqv','dqc','dqi','dqs','dth')
    arrays={n:fields[n] for n in names};first=arrays['dp']
    if first.ndim!=2:raise ValueError('MYNN moisture correction requires (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<1:raise ValueError('MYNN moisture correction needs nonempty columns')
    for name,value in arrays.items():
        _require(name,value,np.float32,(nc,nl))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    params={'nc':np.int32(nc),'nl':np.int32(nl),'delt':np.float32(delt),'xlvcp':np.float32(xlvcp),'xlscp':np.float32(xlscp)}
    _launch('hexv842_mpas_bl_mynn_moisture_f32',nc,context,arrays,params)


def tendencies(fields, *, constants, flags, rp7=False):
    """Native MYNN tendency update with device packing and internal solves.

    Inputs are source-valid independent column rows with distinct storage,
    positive depths and density, and nonzero timesteps and denominators.
    Only source-written profile arrays are scattered back to the caller.
    """
    from ..metal_dynamics_v841 import _arrays
    profiles=('dz','rho','u','v','th','tk','qv','qc','qi','qs','qnc','qni','p','exner','thl','sqv','sqc','sqi','sqs','sqw','qnwfa','qnifa','qnbca','ozone','tsq','qsq','cov','tcd','qcd','dfm','dfh','dfq','du','dv','dth','dqv','dqc','dqi','dqs','dqnc','dqni','dqnwfa','dqnifa','dqnbca','dozone','diss_heat','s_aw','s_awthl','s_awqt','s_awqv','s_awqc','s_awu','s_awv','s_awqnc','s_awqni','s_awqnwfa','s_awqnifa','s_awqnbca','sd_aw','sd_awthl','sd_awqt','sd_awqv','sd_awqc','sd_awu','sd_awv','sub_thl','sub_sqv','sub_u','sub_v','det_thl','det_sqv','det_sqc','det_u','det_v','cldfra_bl1d')
    outputs=('thl','du','dv','dth','dqv','dqc','dqi','dqs','dqnc','dqni','dqnwfa','dqnifa','dqnbca','dozone')
    columns=('psfc','ust','flt','flq','flqv','flqc','wspd','uoce','voce')
    constnames=('delt','r_d','p608','xlvcp','xlscp','t0c')
    flagnames=('flag_qc','flag_qi','flag_qnc','flag_qni','flag_qs','flag_qnwfa','flag_qnifa','flag_qnbca','flag_ozone','bl_mynn_cloudmix','bl_mynn_mixqt','bl_mynn_edmf_mom','bl_mynn_mixscalars')
    arrays={n:fields[n] for n in profiles+columns};first=arrays['dz']
    if first.ndim!=2:raise ValueError('MYNN tendencies require (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<2:raise ValueError('MYNN tendencies need columns and at least two levels')
    for name,value in arrays.items():
        _require(name,value,np.float32,(nc,) if name in columns else (nc,nl))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    if 4*nc*(len(profiles)*nl+28*(nl+1))>context.recommended_working_set_size:raise MemoryError('MYNN tendencies workspace exceeds Metal working set')
    xp=_arrays(context);packed=xp.empty((len(profiles),nc,nl),np.float32);scratch=xp.empty((nc,28,nl+1),np.float32)
    dims={'nc':np.int32(nc),'nl':np.int32(nl)}
    for j,names in enumerate((profiles[:40],profiles[40:])):
        _launch(f'hexv842_mpas_bl_mynn_tendencies_pack{j}_f32',nc*nl,context,{'packed':packed}|{n:arrays[n] for n in names},dims)
    params=dims|{n:np.float32(constants[n]) for n in constnames}|{n:np.int32(flags[n]) for n in flagnames}
    kernel='hexv842_mpas_bl_mynn_tendencies_correct_f32' if rp7 else 'hexv842_mpas_bl_mynn_tendencies_f32'
    if rp7:params['gravity']=np.float32(constants['gravity'])
    _launch(kernel,nc,context,{'packed':packed,'scratch':scratch}|{n:arrays[n] for n in columns},params)
    _launch('hexv842_mpas_bl_mynn_tendencies_scatter_f32',nc*nl,context,{'packed':packed}|{n:arrays[n] for n in outputs},dims)


def edmf_condensation(fields, *, constants):
    """Native MYNN plume condensation for source-valid physical inputs.

    The caller supplies positive-normal pressure divided by p1000mb and
    distinct arrays; outputs are thv and qc.
    """
    names=('qt','thl','p','zagl','thv','qc');arrays={n:fields[n] for n in names};first=arrays['qt']
    if first.ndim!=1:raise ValueError('MYNN plume condensation inputs must be one-dimensional')
    count=int(first.size);context=_context(first)
    for name,value in arrays.items():
        _require(name,value,np.float32,(count,))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    params={'count':np.int32(count)}|{n:np.float32(constants[n]) for n in ('p1000mb','rcp','xlvcp','t0c','rvovrd')}
    if count:_launch('hexv842_mpas_bl_mynn_edmf_condensation_f32',count,context,arrays,params)


def dmp(fields, *, constants, options):
    """Native MYNN eight-plume mass flux with chemistry disabled.

    The caller supplies distinct, source-valid native profiles and physical
    constants, including valid positive-normal inputs to the power operation.
    Interface carriers and zw have one more level than cell-centred profiles.
    The initial height scan must encounter a level at or below 50 m before
    exceeding pblh+500; the native source otherwise reads an unset k50.
    """
    from ..metal_dynamics_v841 import _arrays
    profiles=('zw','dz','p','rho','u','v','w','th','thl','thv','tk','qt','qv','qc','qke','qnc','qni','qnwfa','qnifa','qnbca','exner','vt','vq','sgm','edmf_a','edmf_w','edmf_qt','edmf_thl','edmf_ent','edmf_qc','s_aw','s_awthl','s_awqt','s_awqv','s_awqc','s_awu','s_awv','s_awqke','s_awqnc','s_awqni','s_awqnwfa','s_awqnifa','s_awqnbca','sub_thl','sub_sqv','sub_u','sub_v','det_thl','det_sqv','det_sqc','det_u','det_v','qc_bl1d','cldfra_bl1d','qc_bl1d_old','cldfra_bl1d_old','rstoch_col')
    interfaces={'zw'}|{n for n in profiles if n.startswith('s_aw')}
    outputs=('vt','vq','edmf_a','edmf_w','edmf_qt','edmf_thl','edmf_ent','edmf_qc','s_aw','s_awthl','s_awqt','s_awqv','s_awqc','s_awu','s_awv','s_awqke','s_awqnc','s_awqni','s_awqnwfa','s_awqnifa','s_awqnbca','sub_thl','sub_sqv','sub_u','sub_v','det_thl','det_sqv','det_sqc','det_u','det_v','qc_bl1d','cldfra_bl1d')
    columns=('ust','flt','fltv','flq','flqv','pblh','dx','landsea','ts','psig_shcu')
    outcolumns=('maxwidth','maxmf','ztop','ktop')
    constnames=('dt','p1000mb','rcp','xlvcp','t0c','rvovrd','r_v','ep_2','ep_3','cice','cliq','cpv','xls','xlv','cp','p608','gravity','ep1')
    flags=('momentum_opt','tke_opt','scalar_opt')
    arrays={n:fields[n] for n in profiles+columns+outcolumns+('kpbl',)};first=arrays['dz']
    if first.ndim!=2:raise ValueError('MYNN plume fields require (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<3:raise ValueError('MYNN plume calculation needs columns and at least three levels')
    for name,value in arrays.items():
        shape=(nc,) if name in columns+outcolumns+('kpbl',) else (nc,nl+1) if name in interfaces else (nc,nl)
        _require(name,value,np.int32 if name in ('kpbl','ktop') else np.float32,shape)
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    if 4*nc*(len(profiles)+150+8)*(nl+1)>context.recommended_working_set_size:raise MemoryError('MYNN plume workspace exceeds Metal working set')
    xp=_arrays(context);packed=xp.empty((len(profiles),nc,nl+1),np.float32);scratch=xp.empty((nc,150,nl+1),np.float32);intscratch=xp.empty((nc,8,nl+1),np.int32)
    dims={'nc':np.int32(nc),'nl':np.int32(nl)}
    for i,names in enumerate((profiles[:25],profiles[25:50],profiles[50:])):
        _launch(f'hexv842_mpas_bl_mynn_dmp_pack{i}_f32',nc*(nl+1),context,{'packed':packed}|{n:arrays[n] for n in names},dims)
    params=dims|{n:np.float32(constants[n]) for n in constnames}|{n:np.int32(options[n]) for n in flags}
    _launch('hexv842_mpas_bl_mynn_dmp_f32',nc,context,{'packed':packed,'scratch':scratch,'intscratch':intscratch}|{n:arrays[n] for n in columns+outcolumns+('kpbl',)},params)
    _launch('hexv842_mpas_bl_mynn_dmp_scatter_f32',nc*(nl+1),context,{'packed':packed}|{n:arrays[n] for n in outputs},dims)


def predict_columns(fields, *, closure, delt, karman, edmf_tke, tkebudget):
    """Native predictor with actual per-column surface values and interfaces.

    The caller supplies independent contiguous rows, positive layer depths
    and density, and nonzero surface scales.
    """
    from ..metal_dynamics_v841 import _arrays
    if tkebudget:raise ValueError('MYNN TKE budget reads an uninitialized top-level bp in the native source')
    profiles=('dz','el','dfq','rho','pdk','pdt','pdq','pdc','qke','tsq','qsq','cov','qwt1d','qdiss1d')
    interfaces=('s_aw','s_awqke');columns=('ust','flt','flq','pmz','phh')
    arrays={n:fields[n] for n in profiles+interfaces+columns};first=arrays['dz']
    if first.ndim!=2:raise ValueError('MYNN predictor fields require (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<2:raise ValueError('MYNN predictor needs columns and at least two levels')
    for name,value in arrays.items():
        shape=(nc,) if name in columns else (nc,nl+1) if name in interfaces else (nc,nl)
        _require(name,value,np.float32,shape)
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    if 4*nc*18*(nl+1)>context.recommended_working_set_size:raise MemoryError('MYNN predictor scratch exceeds Metal working set')
    arrays['scratch']=_arrays(context).empty((nc,18,nl+1),np.float32)
    params={'nc':np.int32(nc),'nl':np.int32(nl),'closure':np.float32(closure),'delt':np.float32(delt),'karman':np.float32(karman),'bl_mynn_edmf_tke':np.int32(edmf_tke),'bl_mynn_tkebudget':np.int32(tkebudget)}
    _launch('hexv842_mpas_bl_mynn_predict_columns_f32',nc,context,arrays,params)


def driver_pre(fields, *, xlvcp, xlscp, cold):
    """Native MYNN column preparation for cold or warm non-cycling calls.

    Caller owns distinct column-major buffers, valid Exner and specific vapor,
    and supplies persistent fields in scratch storage before publication.
    """
    inputs=('dz1','th1','sqv1','sqc1','sqi1','sqs1','ex1')
    outputs=('qv1','qc1','qi1','qs1','sqw','thl','thetav','qke1','sh1','sm1','el_pbl1','tsq1','qsq1','cov1','cldfra_bl1','qc_bl1','qi_bl1')
    arrays={n:fields[n] for n in inputs+outputs+('ust1','zw')};first=arrays['dz1']
    if first.ndim!=2:raise ValueError('MYNN driver preparation requires (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<2:raise ValueError('MYNN driver preparation needs columns and at least two levels')
    for name,value in arrays.items():
        shape=(nc,) if name=='ust1' else (nc,nl+1) if name=='zw' else (nc,nl)
        _require(name,value,np.float32,shape)
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    _launch('hexv842_mpas_bl_mynn_driver_pre_f32',nc,context,arrays,{'nc':np.int32(nc),'nl':np.int32(nl),'xlvcp':np.float32(xlvcp),'xlscp':np.float32(xlscp),'cold':np.int32(cold)})


def driver_surface(fields, *, constants):
    """Native per-column surface controls for MYNN stability selector 1.

    Caller supplies independent, source-valid fields, including positive-normal
    ps1/p1000mb, positive density, nonzero Exner and physical surface scales.
    """
    profiles=('qv1','rho1','ex1','dz1');columns=('ps1','ts1','ust1','hfx1','qfx1')
    outputs=('flqv','flqc','th_sfc','flq','flt','fltv','rmol1','zet','pmz','phh')
    constnames=('cp','p1000mb','rcp','xlvcp','p608','karman','gravity')
    arrays={n:fields[n] for n in profiles+columns+outputs};first=arrays['qv1']
    if first.ndim!=2:raise ValueError('MYNN driver surface inputs require (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<1:raise ValueError('MYNN driver surface needs nonempty columns')
    for name,value in arrays.items():
        _require(name,value,np.float32,(nc,nl) if name in profiles else (nc,))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    _launch('hexv842_mpas_bl_mynn_driver_surface_f32',nc,context,arrays,{'nc':np.int32(nc),'nl':np.int32(nl)}|{n:np.float32(constants[n]) for n in constnames})


def driver_heat(fields, *, cp):
    """Native default dissipation heating for positive-normal qke inputs.

    Caller supplies independent column rows, positive-normal qke, and valid
    pressure, length, and heat-capacity inputs before publishing results.
    The native predictor supplies qke bounded to [1e-4, 150].
    """
    names=('qke1','el_pbl1','p1','diss_heat')
    arrays={n:fields[n] for n in names};first=arrays['qke1']
    if first.ndim!=2:raise ValueError('MYNN driver heat requires (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<2:raise ValueError('MYNN driver heat needs columns and at least two levels')
    for name,value in arrays.items():
        _require(name,value,np.float32,(nc,nl))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    _launch('hexv842_mpas_bl_mynn_driver_heat_f32',nc,context,arrays,{'nc':np.int32(nc),'nl':np.int32(nl),'cp':np.float32(cp)})


def driver_exchange(fields):
    """Native exchange coefficients from independent column profiles."""
    names=('dfm','dfh','dz','k_m','k_h')
    arrays={n:fields[n] for n in names};first=arrays['dz']
    if first.ndim!=2:raise ValueError('MYNN exchange requires (column,level)')
    nc,nl=map(int,first.shape);context=_context(first)
    if nc<1 or nl<1:raise ValueError('MYNN exchange needs nonempty columns')
    for name,value in arrays.items():
        _require(name,value,np.float32,(nc,nl))
        if _context(value) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    _launch('hexv842_mpas_bl_mynn_driver_exchange_f32',nc,context,arrays,{'nc':np.int32(nc),'nl':np.int32(nl)})


def interface_prefix(source, target):
    """Copy each native (nl+1) interface row into its nl-level view."""
    if target.ndim!=2:raise ValueError('MYNN target requires (column,level)')
    nc,nl=map(int,target.shape);context=_context(target)
    if nc<1 or nl<1:raise ValueError('MYNN interface copy needs nonempty columns')
    _require('source',source,np.float32,(nc,nl+1));_require('target',target,np.float32,(nc,nl))
    if _context(source) is not context:raise ValueError('MYNN arrays belong to different Metal contexts')
    _launch('hexv842_mpas_bl_mynn_driver_interface_f32',nc*nl,context,{'input':source,'output':target},{'nc':np.int32(nc),'nl':np.int32(nl)})
