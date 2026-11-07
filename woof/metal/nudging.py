"""Regional FDDA with source arithmetic and host-staged horizontal FFTs.

As for host stochastic patterns, NumPy's transform defines the host reference;
this does not claim cuFFT word identity. Physical-temperature pressure powers
use the host libm that also builds the CUDA reference; the remaining field
arithmetic runs in Metal with source rounding.
"""
from types import SimpleNamespace
import ctypes
import math
import numpy as np

from .array import MetalArray

_POWF = ctypes.CDLL(None).powf
_POWF.argtypes = (ctypes.c_float, ctypes.c_float)
_POWF.restype = ctypes.c_float


def submit(xp, name, resources, parameters, size):
    """Submit the flat FDDA ABI, including complex and binary64 storage."""
    if name in ('pack_fields', 'unpack_fields'):
        resources = dict(resources)
        resources['exner'] = resources['p']
        if not parameters['potential']:
            pressure = resources['p'].to_numpy() / np.float32(100000.)
            exponent = float(np.float32(287.) / np.float32(1004.5))
            factor = np.fromiter((_POWF(float(p), exponent) for p in pressure.flat),
                                 dtype=np.float32, count=pressure.size)
            resources['exner'] = xp.asarray(factor.reshape(pressure.shape))
    resources = {key: value.view((value.nbytes // 4,), dtype=np.float32)
                 if value.dtype == np.dtype(np.complex64) else
                 value.view((value.size,)) for key, value in resources.items()}
    parameters = dict(parameters)
    return xp.context.submit_kernel('nudging_'+name, resources, parameters,
                                   exact_policy=False, grid_size=int(size))


def storage(xp, state, pbl_height, potential):
    nz, ny, nx = state.thp.shape
    shapes = dict(u=(nz,ny,nx+1), v=(nz,ny+1,nx),
                  temperature=(nz,ny,nx), geopotential=(nz+1,ny,nx))
    physical = {name: getattr(state,name) if name in ('u','v') else xp.empty(shape)
                for name,shape in shapes.items()}
    previous = {name: xp.empty(shapes[name]) for name in ('temperature','geopotential')}
    heights = {name: xp.empty(shape) for name,shape in shapes.items()}
    pbl = {name: xp.empty(shape[-2:]) for name,shape in shapes.items()}
    resources = dict(thp=state.thp, thb=state.thb, p=state.p,
                     php=state.php, phb=state.phb, ht=state.ht, pbl=pbl_height,
                     temp=physical['temperature'], oldt=previous['temperature'],
                     phi=physical['geopotential'], oldphi=previous['geopotential'])
    resources.update(zip(('zu','zv','zt','zf'),heights.values()))
    resources.update(zip(('bu','bv','bt','bf'),pbl.values()))
    params = dict(nz=nz,ny=ny,nx=nx,terrain=int(state.thb.ndim==3),potential=int(potential))
    def pack():
        submit(xp,'pack_fields',resources,params,max(np.prod(s) for s in shapes.values()))
    return physical,previous,heights,pbl,params,pack


def analysis_fields(xp, host, convention):
    """Pack prepared endpoints with the same physical-carrier kernel."""
    values = {name:xp.asarray(a,dtype=np.float32) for name,a in host.items()}
    nz,ny,nx = values['thp'].shape
    state = SimpleNamespace(**values, p=values['pressure'], ht=xp.zeros((ny,nx)))
    fields,_,_,_,_,pack = storage(xp,state,xp.zeros((ny,nx)),convention=='wrf')
    pack()
    fields['qv']=values['qv']
    return fields


class MetalNudger:
    """One forecast operator at the existing post-RK nudging slot."""
    def __init__(self,state,run_config,config,analyses,*,interval_start,
                 interval_end,pbl_height=None):
        self.state,self.config,self.xp = state,config,state._array_module
        xp=self.xp
        if run_config.use_adaptive_time_step:
            raise ValueError('spectral nudging requires fast numerics and a fixed time step')
        from woof.wrf_exact import DIAGNOSTICS_ENABLED
        if DIAGNOSTICS_ENABLED:
            raise ValueError('spectral nudging requires fast numerics and a fixed time step')
        if getattr(state,'_spectral_nudging_graph',None) is not None:
            raise ValueError('nudging already attached; replace its forcing interval at a seam')
        if not (run_config.specified or run_config.nested):
            raise ValueError('NUDGING_REGIONAL_GEOMETRY: nudging needs a specified root or a fixed regional nest')
        if config.convention=='regional' and config.boundary_cells < run_config.spec_zone+run_config.relax_zone:
            raise ValueError('nudging boundary_cells must protect the full lateral relaxation zone')
        nz,ny,nx=state.thp.shape
        if pbl_height is None:
            pbl_height=getattr(getattr(state,'physics',None),'fields',{}).get('pblh')
        if (pbl_height is None and config.convention=='wrf' and
                not any(getattr(config,'if_no_pbl_nudging_'+s) for s in ('uv','t','ph','q'))):
            pbl_height=xp.zeros((ny,nx))
        if (not isinstance(pbl_height,MetalArray) or pbl_height.shape!=(ny,nx)
                or pbl_height.dtype!=np.dtype(np.float32)):
            raise ValueError('bind a live float32 mass-grid PBL height buffer')
        self.pack_pbl=pbl_height
        self.physical,self.previous,self.heights,self.pbl,self.geometry,self.pack = storage(xp,state,pbl_height,config.convention=='wrf')
        if config.coefficient('qv')>0:
            self.physical['qv']=state.qv
            self.heights['qv']=self.heights['temperature']
            self.pbl['qv']=self.pbl['temperature']
        self.weights={name:xp.empty(a.shape) for name,a in self.physical.items()} if config.convention=='wrf' else {}
        self.bindings={name:a for name,a in self.physical.items() if config.coefficient(name)>0}
        self.scratch={}
        self.held={}
        for name,value in self.bindings.items():
            config.validate_grid(value.shape)
            self.scratch[name]=xp.zeros(value.shape,dtype=np.complex64)
            carried=getattr(state,'nudging_tendency_'+name,None)
            if config.cadence_steps(run_config.dt)>1 and carried is None:
                raise ValueError('fgdt requires restart-carried held tendencies')
            self.held[name]=carried if carried is not None else xp.zeros(value.shape)
        self.timing=xp.asarray(np.array([-1. if config.gfdda_end_h is None else config.gfdda_end_h*3600.,config.dtramp_min*60. if config.if_ramping else 0.],dtype=np.float64))
        self.clock=xp.empty((4,),dtype=np.float64)
        self.error=xp.zeros((1,),dtype=np.uint32)
        self.stage_count=xp.zeros((1,),dtype=np.uint64)
        self.replace_interval(analyses,elapsed_seconds=state.elapsed_seconds,
            dt_seconds=run_config.dt,interval_start=interval_start,interval_end=interval_end)
        self.graph=self
        state._spectral_nudging_graph=self
        state._spectral_nudging_owner=self

    def replace_interval(self,analyses,*,elapsed_seconds,dt_seconds,interval_start,interval_end):
        cfg=self.config
        stop=None if cfg.gfdda_end_h is None else cfg.gfdda_end_h*3600.
        if (not all(math.isfinite(v) for v in (elapsed_seconds,dt_seconds,interval_start,interval_end))
                or not 0<dt_seconds<=cfg.relaxation_seconds or interval_start>=interval_end
                or elapsed_seconds<interval_start or
                ((stop is None or elapsed_seconds<stop) and elapsed_seconds+dt_seconds>interval_end)):
            raise ValueError('forcing must bracket the step and 0 < dt <= relaxation_seconds')
        if set(analyses)!=set(self.bindings):
            raise ValueError('a forcing interval must replace every selected field')
        for name,pair in analyses.items():
            if len(pair)!=2 or any(not isinstance(a,MetalArray) or a.shape!=self.bindings[name].shape
                    or a.dtype!=np.dtype(np.float32) or a.buffer.context is not self.xp.context for a in pair):
                raise ValueError(f'invalid analysis endpoint for {name}')
        self.check_at_seam()
        self.analyses=analyses
        self.dt=dt_seconds
        self.clock[...]=np.array([elapsed_seconds,dt_seconds,interval_start,interval_end],dtype=np.float64)

    def check_at_seam(self):
        if int(self.error.to_numpy()[0]):
            raise RuntimeError('spectral nudging step escaped its analysis interval; state was not nudged')

    def launch(self):
        cfg,xp=self.config,self.xp
        self.pack()
        nz,ny,nx=self.state.thp.shape
        for name,w in self.weights.items():
            field=('u','v','temperature','geopotential','qv').index(name)
            suffix={'u':'uv','v':'uv','temperature':'t','geopotential':'ph','qv':'q'}[name]
            params={k:v for k,v in self.geometry.items() if k!='potential'}
            params.update(field=field,grid_nudging=int(cfg.mode=='grid'),
                no_pbl=int(getattr(cfg,'if_no_pbl_nudging_'+suffix)),
                zfac=int(getattr(cfg,'if_zfac_'+suffix)),kz=getattr(cfg,'k_zfac_'+suffix),
                dk=getattr(cfg,'dk_zfac_'+suffix))
            submit(xp,'wrf_weights',dict(php=self.state.php,phb=self.state.phb,
                ht=self.state.ht,pbl=self.pack_pbl,weight=w),params,w.shape[-2]*w.shape[-1])
        common=dict(wrf=int(cfg.convention=='wrf'),cadence=cfg.cadence_steps(self.dt))
        for name,value in self.bindings.items():
            work=self.scratch[name]
            a,b=self.analyses[name]
            submit(xp,'difference',dict(value=value,a=a,b=b,out=work,clock=self.clock,error=self.error,timing=self.timing),
                dict(common,n=value.size,grid_nudging=int(cfg.mode=='grid')),value.size)
            sy,sx=value.shape[-2:]
            if cfg.mode=='spectral':
                spectrum=xp.asarray(np.fft.fft2(work.to_numpy(),axes=(-2,-1)).astype(np.complex64))
                submit(xp,'truncate_modes',dict(spectrum=spectrum),dict(n=value.size,nx=sx,ny=sy,
                    cutoff_x=cfg.cutoff_x,cutoff_y=cfg.cutoff_y),value.size)
                work=xp.asarray(np.fft.ifft2(spectrum.to_numpy(),axes=(-2,-1)).astype(np.complex64))
            submit(xp,'relax',dict(value=value,filtered=work,height_agl=self.heights[name],pblh=self.pbl[name],
                clock=self.clock,timing=self.timing,vertical_weight=self.weights.get(name,self.heights[name]),held=self.held[name]),
                dict(common,n=value.size,nx=sx,ny=sy,boundary=cfg.boundary_cells,
                     floor_m=cfg.minimum_height_m,ramp_m=cfg.ramp_depth_m,tau=1./cfg.coefficient(name),
                     fgdtzero=int(cfg.fgdtzero)),value.size)
        params={k:v for k,v in self.geometry.items() if k!='terrain'}
        submit(xp,'unpack_fields',dict(thp=self.state.thp,php=self.state.php,p=self.state.p,
            temp=self.physical['temperature'],oldt=self.previous['temperature'],
            phi=self.physical['geopotential'],oldphi=self.previous['geopotential']),params,(nz+1)*ny*nx)
        submit(xp,'tick',dict(clock=self.clock,error=self.error,count=self.stage_count),dict(n=1),1)
