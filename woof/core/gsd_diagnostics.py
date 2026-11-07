"""Operational GSD diagnostics, NOAA-EMC/HRRR 40ee6058 (WRFV3.9).

Every reduction remains on the device. History reads do not reset these
windows. See docs/dev/operational-output-controls.md for source and limits.
"""
from __future__ import annotations
import numpy as np

HISTORY = {}
for names, units, description in (
    (("wspd10max","wspd10umax","wspd10vmax","wspd80max","wspd80umax","wspd80vmax"), "m s-1", "Wind at interval maximum speed"),
    (("w_up_max","w_dn_max","w_mean","wspd10","wspd80"), "m s-1", "GSD vertical or horizontal wind diagnostic"),
    (("grpl_max",), "kg m-2", "Maximum column integrated graupel"),
    (("refd_max","refdm10c_max"), "dbZ", "Maximum derived radar reflectivity"),
    (("composite_refl_10cm","refl_10cm_1km","refl_10cm_4km"), "dbZ", "GSD reflectivity at history time"),
    (("ltg1_max","ltg2_max","ltg3_max"), "flashes km-2 5min-1", "Maximum lightning threat"),
    (("lwp","iwp"), "g m-2", "Column cloud water path"),
    (("maxcldfra",), "0 to 1", "Maximum boundary layer cloud fraction"),
    (("rel_vort","rel_vort_max","rel_vort_max01"), "s-1", "Relative vertical vorticity"),
):
    for slot in names:
        symbol = slot if slot.startswith("rel_vort") else slot.upper()
        HISTORY[slot] = (symbol, units, description)
BANDS = (("",2000.,5000.),("16",1000.,6000.),("02",0.,2000.),("03",0.,3000.))
for suffix, lower, upper in BANDS:
    for prefix in ("up_heli_max", "up_heli_min", "uh"):
        slot=prefix+suffix
        HISTORY[slot]=(slot.upper(),"m2 s-2",f"Updraft helicity {lower:g}-{upper:g} m AGL")
HISTORY["up_heli_max"]=("UP_HELI_MAX","m2 s-2","MAX UPDRAFT HELICITY")
PERSISTENT = (*(slot for slot in HISTORY if slot != "up_heli_max"), "gsd_w_mean_sum", "gsd_step_count", "gsd_ltg1", "gsd_ltg2", "gsd_total_steps")
WORK = ("gsd_zero", "gsd_uh_use", "gsd_pressure_w", "gsd_temperature",
        "gsd_density", "gsd_wind10", "gsd_wind80")

def _is_metal(array):
    try:
        from woof.metal.array import MetalArray
    except ImportError:  # CUDA-only installations
        return False
    return isinstance(array, MetalArray)

def _boundary_copy(target):
    if _is_metal(target):
        from woof.metal.gsd_diagnostics_family import copy_inner_boundary
        copy_inner_boundary(target)
        return
    target[:,0]=target[:,1];target[0,:]=target[1,:]

def scratch_shapes(cfg):
    if not cfg.gsd_diagnostics:
        return {}
    plane=(cfg.ny,cfg.nx); mass=(cfg.nz,*plane)
    shapes={slot: mass if slot in ("rel_vort","gsd_zero","gsd_temperature","gsd_density") else plane
            for slot in (*PERSISTENT,*WORK)}
    shapes["gsd_pressure_w"]=(cfg.nz+1,*plane)
    return shapes

def initialize(state,cfg):
    for slot,shape in scratch_shapes(cfg).items():
        state.scratch(shape,slot)
    if cfg.gsd_diagnostics:
        for name in ("wspd10","wspd80"):
            state.existing_scratch(name)[...]=-99.

def update_helicity(state,cfg,*,history_due=False):
    """Sample start-of-step state, as module_first_rk_step_part2 does."""
    if not cfg.gsd_diagnostics:
        return
    from woof.core.uh_diag import _supported_boundary_geometry
    if not _supported_boundary_geometry(cfg):
        raise NotImplementedError("gsd_diagnostics requires specified/nested or open x/y boundaries")
    nz,ny,nx=cfg.nz,cfg.ny,cfg.nx
    if nx<4 or ny<4 or nz<3:
        raise ValueError("gsd_diagnostics requires nx, ny >= 4 and nz >= 3 for helicity stencils")
    metal=_is_metal(state.p)
    f=np.float32;i=np.int32
    if metal:
        from woof.metal.gsd_diagnostics_family import submit_gsd
    else:
        from woof.core.kernels import get_kernel
        grid=((nx*ny+127)//128,); block=(128,)
    zero=state.existing_scratch("gsd_zero")
    scratch=state.existing_scratch
    rv=scratch("rel_vort")
    idump=max(1,int(f(cfg.reset_interval1*60)/f(cfg.dt)))
    rdx=f(1)/f(cfg.dx);rdy=f(1)/f(cfg.dy)
    if metal:
        submit_gsd("gsd_prepare",
            dict(u=state.u,v=state.v,ph=state.php,phb=state.phb,thp=state.thp,thb=state.thb,
                 p=state.p,alt=state.alt,qv=state.qv if state.qv is not None else zero,
                 fnm=state.fnm,fnp=state.fnp,pw=scratch("gsd_pressure_w"),
                 temperature=scratch("gsd_temperature"),rho=scratch("gsd_density"),
                 wind10=scratch("gsd_wind10"),wind80=scratch("gsd_wind80")),
            dict(phb3=int(state.phb.ndim==3),thb3=int(state.thb.ndim==3),nx=nx,ny=ny,nz=nz,
                 diagflag=int(bool(history_due))),nx*ny)
        submit_gsd("gsd_vorticity",
            dict(u=state.u,v=state.v,ph=state.php,phb=state.phb,msfu=state.msfu,msfv=state.msfv,
                 dn=state.dn,dnw=state.dnw,fnm=state.fnm,fnp=state.fnp,rv=rv),
            dict(phb3d=int(state.phb.ndim==3),cf1=state.cf1,cf2=state.cf2,cf3=state.cf3,
                 rdx=rdx,rdy=rdy,nz=nz,ny=ny,nx=nx),nz*ny*nx)
        for suffix,lower,upper in BANDS:
            column=scratch("uh"+suffix);use=scratch("gsd_uh_use")
            submit_gsd("gsd_uh_columns",
                dict(w=state.w,ph=state.php,phb=state.phb,ht=state.ht,rv=rv,uh=column,usecol=use),
                dict(phb3d=int(state.phb.ndim==3),lower=lower,upper=upper,nz=nz,ny=ny,nx=nx),nx*ny)
            submit_gsd("gsd_uh_fold",
                dict(uh=column,use=use,maximum=scratch("up_heli_max"+suffix),
                     minimum=scratch("up_heli_min"+suffix)),dict(ny=ny,nx=nx),nx*ny)
            for slot in ("up_heli_max"+suffix,"up_heli_min"+suffix):
                _boundary_copy(scratch(slot))
        submit_gsd("gsd_vorticity_fold",
            dict(rv=rv,ph=state.php,phb=state.phb,ht=state.ht,maximum=scratch("rel_vort_max"),
                 max01=scratch("rel_vort_max01"),total_steps=scratch("gsd_total_steps")),
            dict(phb3d=int(state.phb.ndim==3),idump=idump,nz=nz,ny=ny,nx=nx),nx*ny)
        for slot in ("rel_vort_max", "rel_vort_max01"):
            _boundary_copy(scratch(slot))
        return
    u, v = state.u, state.v
    if getattr(state, "_moving_frame_velocity", None):
        from woof.moving_frame import ground_relative_winds
        u, v = ground_relative_winds(state)
    get_kernel("gsd_diagnostics","gsd_prepare")(grid,block,
        (u,v,state.php,state.phb,i(state.phb.ndim==3),state.thp,
         state.thb,i(state.thb.ndim==3),state.p,state.alt,
         state.qv if state.qv is not None else zero,state.fnm,state.fnp,
         *(state.existing_scratch(name) for name in
           ("gsd_pressure_w","gsd_temperature","gsd_density","gsd_wind10","gsd_wind80")),
         i(nx),i(ny),i(nz),i(bool(history_due))))
    get_kernel("gsd_helicity","gsd_vorticity")(((nz*ny*nx+127)//128,),block,
        (state.u,state.v,state.php,state.phb,i(state.phb.ndim==3),
         state.msfu,state.msfv,state.dn,state.dnw,state.fnm,state.fnp,
         f(state.cf1),f(state.cf2),f(state.cf3),rdx,rdy,
         rv,i(nz),i(ny),i(nx)))
    for suffix,lower,upper in BANDS:
        column=state.existing_scratch("uh"+suffix)
        use=state.existing_scratch("gsd_uh_use")
        get_kernel("gsd_helicity","gsd_uh_columns")(grid,block,
            (state.w,state.php,state.phb,i(state.phb.ndim==3),state.ht,rv,
             column,use,f(lower),f(upper),i(nz),i(ny),i(nx)))
        get_kernel("gsd_helicity","gsd_uh_fold")(grid,block,
            (column,use,state.existing_scratch("up_heli_max"+suffix),
             state.existing_scratch("up_heli_min"+suffix),i(ny),i(nx)))
        for slot in ("up_heli_max"+suffix,"up_heli_min"+suffix):
            _boundary_copy(state.existing_scratch(slot))
    get_kernel("gsd_helicity","gsd_vorticity_fold")(grid,block,
        (rv,state.php,state.phb,i(state.phb.ndim==3),state.ht,
         state.existing_scratch("rel_vort_max"),state.existing_scratch("rel_vort_max01"),
         state.existing_scratch("gsd_total_steps"),
         i(idump),i(nz),i(ny),i(nx)))
    for slot in ("rel_vort_max", "rel_vort_max01"):
        _boundary_copy(state.existing_scratch(slot))

def reset_interval(state,cfg):
    if not cfg.gsd_diagnostics:
        return
    from woof.core.output_diagnostics import boundary_due
    if not boundary_due(state.elapsed_seconds,cfg.reset_interval1*60.):
        return
    # In the fork this reset follows the first-RK helicity calculation.
    for slot in (*PERSISTENT, "up_heli_max"):
        if slot.startswith("uh") or slot in ("rel_vort","gsd_ltg1","gsd_ltg2",
                "gsd_total_steps","composite_refl_10cm","refl_10cm_1km","refl_10cm_4km"):
            continue
        state.existing_scratch(slot)[...]=-99. if slot in ("wspd10","wspd80") else 0.
    for slot in ("hail_maxk1","hail_max2d"):
        value=state.existing_scratch(slot)
        if value is not None:
            value[...]=0.

def update(state,cfg,*,history_due):
    if not cfg.gsd_diagnostics:
        return
    nx,ny,nz=cfg.nx,cfg.ny,cfg.nz
    zero=state.existing_scratch("gsd_zero")
    u, v = state.u, state.v
    if getattr(state, "_moving_frame_velocity", None):
        from woof.moving_frame import ground_relative_winds
        u, v = ground_relative_winds(state)
    def species(name):
        value=getattr(state,name,None)
        return zero if value is None else value
    fields=getattr(state.physics,"fields",{})
    def surface(name):
        return fields.get(name,zero[0])
    def cloud(name):
        return fields.get(name,zero)
    scratch=state.existing_scratch
    f=np.float32;i=np.int32
    metal=_is_metal(state.p)
    hail=tuple(scratch(slot) if scratch(slot) is not None else zero[0]
               for slot in ("hail_maxk1","hail_max2d"))
    if metal:
        from woof.metal.gsd_diagnostics_family import submit_gsd
        xp=state._array_module
        arrays=dict(u=state.u,v=state.v,w=state.w,ph=state.php,phb=state.phb,thp=state.thp,
                    thb=state.thb,p=state.p,alt=state.alt,
                    **{name:species(name) for name in ("qv","qc","qr","qi","qs","qg","qh")},
                    znw=state.znw,density=scratch("gsd_density"),temperature=scratch("gsd_temperature"),
                    wind10=scratch("gsd_wind10"),wind80=scratch("gsd_wind80"),msft=state.msft,
                    qcbl=cloud("qc_bl"),cfbl=cloud("cldfra_bl"),u10=surface("u10"),v10=surface("v10"),
                    pw=scratch("gsd_pressure_w"))
        arrays.update(zip(("wspdmax","umax","vmax","wup","wdn","wmean","wsum","count","grplmax",
                           "refmax","refm10","ltg1max","ltg2max","ltg1","ltg2","wspd10","wspd80",
                           "lwp","iwp","maxcf"),(scratch(name) for name in COLUMN_OUTPUTS)))
        arrays.update(hailk1=hail[0],hail2d=hail[1])
        submit_gsd("gsd_columns",arrays,
            dict(phb3=int(state.phb.ndim==3),thb3=int(state.thb.ndim==3),nx=nx,ny=ny,nz=nz,
                 ptop=state.p_top,diagflag=int(bool(history_due)),icloud_bl=cfg.icloud_bl,
                 thompson=int(cfg.mp_physics in (8,28))),nx*ny)
        # Domain maxima are one scalar each; the planes are read back, not extracted per step.
        maxima=[xp.asarray(np.asarray([scratch(slot).to_numpy().max()],dtype=np.float32))
                for slot in ("ltg1_max","ltg2_max")]
        submit_gsd("gsd_lightning3",
            dict(a=scratch("ltg1_max"),b=scratch("ltg2_max"),out=scratch("ltg3_max"),
                 ma=maxima[0],mb=maxima[1]),dict(n=nx*ny),nx*ny)
        for slot in ("hail_maxk1","hail_max2d"):
            if scratch(slot) is not None:
                scratch(slot+"_run")[...]=np.maximum(scratch(slot+"_run").to_numpy(),scratch(slot).to_numpy())
        refl=scratch("refl_10cm")
        if history_due and refl is not None:
            rainc=getattr(state.physics,"cu_raincv",None)
            submit_gsd("gsd_reflectivity",
                dict(refl=refl,phb=state.phb,ht=state.ht,rainc=zero[0] if rainc is None else rainc,
                     composite=scratch("composite_refl_10cm"),at1=scratch("refl_10cm_1km"),
                     at4=scratch("refl_10cm_4km")),
                dict(phb3=int(state.phb.ndim==3),nx=nx,ny=ny,nz=nz,dt=cfg.dt),nx*ny)
        return
    from woof.core.kernels import get_kernel
    import cupy as cp
    get_kernel("gsd_diagnostics","gsd_columns")(((nx*ny+127)//128,),(128,),
        (u,v,state.w,state.php,state.phb,i(state.phb.ndim==3),
         state.thp,state.thb,i(state.thb.ndim==3),state.p,state.alt,
         *(species(name) for name in ("qv","qc","qr","qi","qs","qg","qh")),
         state.znw,scratch("gsd_density"),scratch("gsd_temperature"),
         scratch("gsd_wind10"),scratch("gsd_wind80"),
         state.msft,cloud("qc_bl"),cloud("cldfra_bl"),surface("u10"),surface("v10"),
         scratch("gsd_pressure_w"),
         *(scratch(name) for name in COLUMN_OUTPUTS),
         *hail,
         i(nx),i(ny),i(nz),f(state.p_top),i(bool(history_due)),i(cfg.icloud_bl),
         i(cfg.mp_physics in (8,28))))
    # Multi-pass device reductions, no host scalar extraction or synchronization.
    maxima=tuple(cp.max(scratch(slot)) for slot in ("ltg1_max","ltg2_max"))
    get_kernel("gsd_diagnostics","gsd_lightning3")(((nx*ny+255)//256,),(256,),
        (scratch("ltg1_max"),scratch("ltg2_max"),scratch("ltg3_max"),*maxima,i(nx*ny)))
    for slot in ("hail_maxk1","hail_max2d"):
        if scratch(slot) is not None:
            cp.maximum(scratch(slot+"_run"),scratch(slot),out=scratch(slot+"_run"))
    refl=scratch("refl_10cm")
    if history_due and refl is not None:
        rainc=getattr(state.physics,"cu_raincv",None)
        get_kernel("gsd_diagnostics","gsd_reflectivity")(((nx*ny+127)//128,),(128,),
            (refl,state.phb,i(state.phb.ndim==3),state.ht,
             zero[0] if rainc is None else rainc,
             *(scratch(name) for name in ("composite_refl_10cm","refl_10cm_1km","refl_10cm_4km")),
             i(nx),i(ny),i(nz),f(cfg.dt)))


def finalize_history_fields(fields):
    """Recompute the domain-wide LTG3 ratio after streamed tiles assemble.

    Tile-local ratios cannot represent the native whole-domain reduction.
    Only the two horizontal maxima planes cross the device on a host frame.
    """
    if not all(name in fields for name in ("LTG1_MAX","LTG2_MAX","LTG3_MAX")):
        return fields
    try:
        import cupy as cp
        from woof.core.kernels import get_kernel
    except ImportError:  # no CuPy: the same float32 operations in NumPy
        a,b=(np.asarray(fields[name],dtype=np.float32) for name in ("LTG1_MAX","LTG2_MAX"))
        value=np.float32(.95)*a
        if np.max(b)>0:
            value=value+np.float32(.05)*(np.max(a)/np.max(b))*b
        fields["LTG3_MAX"]=np.where(value<np.float32(.02)*np.float32(1.22),np.float32(0),value).astype(np.float32)
        return fields
    host=isinstance(fields["LTG1_MAX"],np.ndarray)
    a,b=(cp.asarray(fields[name]) for name in ("LTG1_MAX","LTG2_MAX"))
    out=cp.empty_like(a)
    get_kernel("gsd_diagnostics","gsd_lightning3")(((a.size+255)//256,),(256,),
        (a,b,out,cp.max(a),cp.max(b),np.int32(a.size)))
    fields["LTG3_MAX"]=cp.asnumpy(out) if host else out
    return fields

COLUMN_OUTPUTS=("wspd10max","wspd10umax","wspd10vmax","w_up_max","w_dn_max",
    "w_mean","gsd_w_mean_sum","gsd_step_count","grpl_max","refd_max","refdm10c_max",
    "ltg1_max","ltg2_max","gsd_ltg1","gsd_ltg2","wspd10","wspd80","lwp","iwp","maxcldfra")
