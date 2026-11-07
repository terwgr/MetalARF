"""Source-ordered native MYNN column preparation on Metal.

This partial driver covers the default non-cycling cold/warm preparation and
surface controls. It does not publish pools or register the PBL slot.
"""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context
from . import metal_bl_mynn_v842 as ops


def prepare_columns(fields, *, constants, cold, mixlength=2):
    """Run native MYNN preparation through selector-1 surface controls.

    ``fields`` owns independent source-shaped scratch and persistent columns;
    the caller must not publish them until the entire PBL transaction succeeds.
    This is the scale-aware, non-cycling source path. ``cold`` means the source
    condition ``initflag > 0 and not restart``; the caller supplies source-valid
    physical values. Source-initialized EDMF scratch is cleared in place.
    """
    nc,nl=map(int,fields['dz1'].shape);ctx=_context(fields['dz1']);xp=_arrays(ctx)
    fields['edmf_w1'].fill(0);fields['edmf_a1'].fill(0)
    ops.driver_pre(fields,xlvcp=constants['xlvcp'],xlscp=constants['xlscp'],cold=cold)
    zw=xp.empty((nc,nl),np.float32);ops.interface_prefix(fields['zw'],zw)
    def height_and_scale():
        ops.pbl_height(fields['thetav'],fields['qke1'],zw,fields['dz1'],fields['xland1'],fields['pblh1'],fields['kpbl1'])
        zeros=xp.zeros((nc,),np.float32);phim=xp.empty((nc,),np.float32);phih=xp.empty((nc,),np.float32)
        ops.stability(zeros,fields['dx1'],fields['pblh1'],dict(phim=phim,phih=phih,psig_bl=fields['psig_bl'],psig_shcu=fields['psig_shcu']))
    if cold:
        height_and_scale()
        initialize = ops.initialize_length1 if mixlength == 1 else ops.initialize
        initialize(
            dict(dz=fields['dz1'], u=fields['u1'], v=fields['v1'], thl=fields['thl'],
                 qw=fields['sqv1'], theta=fields['th1'], thetav=fields['thetav'],
                 cldfra_bl1d=fields['cldfra_bl1'], edmf_w1=fields['edmf_w1'],
                 edmf_a1=fields['edmf_a1'], sh=fields['sh1'], sm=fields['sm1'],
                 el=fields['el_pbl1'], qke=fields['qke1'], tsq=fields['tsq1'],
                 qsq=fields['qsq1'], cov=fields['cov1'], zw=fields['zw'],
                 zi=fields['pblh1'], rmo=fields['rmol_in'], ust=fields['ust1'],
                 psig_bl=fields['psig_bl']),
            gravity=constants['gravity'], karman=constants['karman'],
            ep1=constants['ep1'], initialize_qke=True)
    height_and_scale()
    ops.driver_surface(fields,constants=constants)


def advance_plumes(fields, *, constants, cold, momentum_opt, tke_opt,
                   scalar_opt, delt, mixlength=2, closure=2.5):
    """Continue the default MYNN path through plume and state prediction.

    Fields are private, independently allocated source columns. This bounded
    path uses cloud PDF 2 and either (mixlength, closure)=(2, 2.5), or
    (1, 2.6) with EDMF momentum enabled. EDMF is on, with no downdraft,
    top-down radiation, stochastic PBL or TKE budget.
    Pool publication, heat, tendencies and diagnostics remain caller-owned.
    """
    nc,nl=map(int,fields['dz1'].shape);ctx=_context(fields['dz1']);xp=_arrays(ctx)
    if ((mixlength, closure) not in ((2, 2.5), (1, 2.6))
            or (mixlength == 1 and not momentum_opt)):
        raise ValueError('unsupported MYNN Metal mixing length, closure or momentum')
    prepare_columns(fields,constants=constants,cold=cold,mixlength=mixlength)
    def profile():return xp.zeros((nc,nl),np.float32)
    def interface():return xp.zeros((nc,nl+1),np.float32)
    def column(dtype=np.float32):return xp.zeros((nc,),dtype)
    s={name:profile() for name in ('vt','vq','sgm','rstoch_col','old_qc','old_cldfra','old_qi','tkeprodtd','dfm','dfh','dfq','tcd','qcd','pdk','pdt','pdq','pdc','qwt','qshear','qbuoy','qdiss','edmf_qt','edmf_thl','edmf_ent','edmf_qc','sub_thl','sub_sqv','sub_u','sub_v','det_thl','det_sqv','det_sqc','det_u','det_v')}
    s.update({name:interface() for name in ('s_aw','s_awthl','s_awqt','s_awqv','s_awqc','s_awu','s_awv','s_awqke','s_awqnc','s_awqni','s_awqnwfa','s_awqnifa','s_awqnbca')})
    s.update(maxwidth=column(),maxmf=column(),ztop=column(),ktop=column(np.int32))
    zw=profile();ops.interface_prefix(fields['zw'],zw)
    ops.condensation2(dict(dz=fields['dz1'],zw=zw,thl=fields['thl'],qw=fields['sqw'],qv=fields['sqv1'],qc=fields['sqc1'],qi=fields['sqi1'],qs=fields['sqs1'],p=fields['p1'],exner=fields['ex1'],tsq=fields['tsq1'],qsq=fields['qsq1'],cov=fields['cov1'],sh=fields['sh1'],el=fields['el_pbl1'],th=fields['th1'],rstoch_col=s['rstoch_col'],qc_bl1d=fields['qc_bl1'],qi_bl1d=fields['qi_bl1'],cldfra_bl1d=fields['cldfra_bl1'],vt=s['vt'],vq=s['vq'],sgm=s['sgm'],xland=fields['xland1'],pblh1=fields['pblh1'],hfx1=fields['hfx1'],rmo=fields['rmol1']),constants=constants)
    plume=dict(zw=fields['zw'],dz=fields['dz1'],p=fields['p1'],rho=fields['rho1'],u=fields['u1'],v=fields['v1'],w=fields['w1'],th=fields['th1'],thl=fields['thl'],thv=fields['thetav'],tk=fields['tk1'],qt=fields['sqw'],qv=fields['sqv1'],qc=fields['sqc1'],qke=fields['qke1'],qnc=fields['qnc1'],qni=fields['qni1'],qnwfa=fields['qnwfa1'],qnifa=fields['qnifa1'],qnbca=fields['qnbca1'],exner=fields['ex1'],vt=s['vt'],vq=s['vq'],sgm=s['sgm'],edmf_a=fields['edmf_a1'],edmf_w=fields['edmf_w1'],qc_bl1d=fields['qc_bl1'],cldfra_bl1d=fields['cldfra_bl1'],qc_bl1d_old=s['old_qc'],cldfra_bl1d_old=s['old_cldfra'],rstoch_col=s['rstoch_col'],ust=fields['ust1'],flt=fields['flt'],fltv=fields['fltv'],flq=fields['flq'],flqv=fields['flqv'],pblh=fields['pblh1'],dx=fields['dx1'],landsea=fields['xland1'],ts=fields['th_sfc'],psig_shcu=fields['psig_shcu'],kpbl=fields['kpbl1'])
    plume.update({n:s[n] for n in ('edmf_qt','edmf_thl','edmf_ent','edmf_qc','s_aw','s_awthl','s_awqt','s_awqv','s_awqc','s_awu','s_awv','s_awqke','s_awqnc','s_awqni','s_awqnwfa','s_awqnifa','s_awqnbca','sub_thl','sub_sqv','sub_u','sub_v','det_thl','det_sqv','det_sqc','det_u','det_v','maxwidth','maxmf','ztop','ktop')})
    ops.dmp(plume,constants=dict(constants,dt=delt),options=dict(momentum_opt=momentum_opt,tke_opt=tke_opt,scalar_opt=scalar_opt))
    turbulence=dict(dz=fields['dz1'],u=fields['u1'],v=fields['v1'],thl=fields['thl'],thetav=fields['thetav'],qw=fields['sqw'],ql=fields['sqc1'],vt=s['vt'],vq=s['vq'],qke=fields['qke1'],tsq=fields['tsq1'],qsq=fields['qsq1'],cov=fields['cov1'],theta=fields['th1'],cldfra_bl1d=fields['cldfra_bl1'],edmf_w1=fields['edmf_w1'],edmf_a1=fields['edmf_a1'],tkeprodtd=s['tkeprodtd'],sh=fields['sh1'],sm=fields['sm1'],el=fields['el_pbl1'],zw=fields['zw'],zi=fields['pblh1'],rmo=fields['rmol1'],fltv=fields['fltv'],psig_bl=fields['psig_bl'])
    turbulence.update({n:s[n] for n in ('dfm','dfh','dfq','tcd','qcd','pdk','pdt','pdq','pdc')});turbulence.update(qshear1d=s['qshear'],qbuoy1d=s['qbuoy'])
    run_turbulence = ops.turbulence26_length1 if mixlength == 1 else ops.turbulence25
    run_turbulence(turbulence, gravity=constants['gravity'],
                   karman=constants['karman'], ep1=constants['ep1'])
    prediction=dict(dz=fields['dz1'],el=fields['el_pbl1'],dfq=s['dfq'],rho=fields['rho1'],qke=fields['qke1'],tsq=fields['tsq1'],qsq=fields['qsq1'],cov=fields['cov1'],s_aw=s['s_aw'],s_awqke=s['s_awqke'],ust=fields['ust1'],flt=fields['flt'],flq=fields['flq'],pmz=fields['pmz'],phh=fields['phh'],qwt1d=s['qwt'],qdiss1d=s['qdiss'])
    prediction.update({n:s[n] for n in ('pdk','pdt','pdq','pdc')})
    ops.predict_columns(prediction,closure=closure,delt=delt,karman=constants['karman'],edmf_tke=tke_opt,tkebudget=False)
    return s


def finish_columns(fields, scratch, *, constants, flags, delt, dheatopt, rp7=False):
    """Run native MYNN heating, tendencies and exchange on private columns.

    Uses the non-downdraft, no-chemistry path from ``advance_plumes``. The
    returned arrays are private source-shaped outputs; pool publication and
    diagnostic option handling must occur after the full transaction succeeds.
    """
    nc,nl=map(int,fields['dz1'].shape);ctx=_context(fields['dz1']);xp=_arrays(ctx)
    def profile():return xp.zeros((nc,nl),np.float32)
    s=scratch
    heat=profile()
    if dheatopt:ops.driver_heat(dict(qke1=fields['qke1'],el_pbl1=fields['el_pbl1'],p1=fields['p1'],diss_heat=heat),cp=constants['cp'])
    kzero=profile();kzero_specific=profile()
    t=dict(dz=fields['dz1'],rho=fields['rho1'],u=fields['u1'],v=fields['v1'],th=fields['th1'],tk=fields['tk1'],qv=fields['qv1'],qc=fields['qc1'],qi=fields['qi1'],qs=kzero,qnc=fields['qnc1'],qni=fields['qni1'],p=fields['p1'],exner=fields['ex1'],thl=fields['thl'],sqv=fields['sqv1'],sqc=fields['sqc1'],sqi=fields['sqi1'],sqs=kzero_specific,sqw=fields['sqw'],qnwfa=fields['qnwfa1'],qnifa=fields['qnifa1'],qnbca=fields['qnbca1'],ozone=fields['qozone1'],tsq=fields['tsq1'],qsq=fields['qsq1'],cov=fields['cov1'],diss_heat=heat,cldfra_bl1d=fields['cldfra_bl1'],psfc=fields['ps1'],ust=fields['ust1'],flt=fields['flt'],flq=fields['flq'],flqv=fields['flqv'],flqc=fields['flqc'],wspd=fields['wspd1'],uoce=fields['uoce1'],voce=fields['voce1'])
    t.update({n:s[n] for n in ('tcd','qcd','dfm','dfh','dfq','sub_thl','sub_sqv','sub_u','sub_v','det_thl','det_sqv','det_sqc','det_u','det_v')})
    for name in ('du','dv','dth','dqv','dqc','dqi','dqs','dqnc','dqni','dqnwfa','dqnifa','dqnbca','dozone','sd_aw','sd_awthl','sd_awqt','sd_awqv','sd_awqc','sd_awu','sd_awv'):t[name]=profile()
    for name in ('s_aw','s_awthl','s_awqt','s_awqv','s_awqc','s_awu','s_awv','s_awqnc','s_awqni','s_awqnwfa','s_awqnifa','s_awqnbca'):
        t[name]=profile();ops.interface_prefix(s[name],t[name])
    ops.tendencies(t,constants=dict(constants,delt=delt),flags=flags,rp7=rp7)
    km,kh=profile(),profile();ops.driver_exchange(dict(dfm=s['dfm'],dfh=s['dfh'],dz=fields['dz1'],k_m=km,k_h=kh))
    return dict(tendencies=t,exch_m=km,exch_h=kh,diss_heat=heat)
