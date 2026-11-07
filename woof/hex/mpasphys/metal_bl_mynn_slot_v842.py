"""Bounded native MPAS v8.4.2 MYNN PBL slot on Metal."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _arrays, _context, _require
from .metal_tendencies_v842 import _launch
from . import metal_bl_mynn_v842 as ops
from .metal_bl_mynn_driver_v842 import advance_plumes, finish_columns
from .bl_mynn_v842 import MynnOptions


def validate_mynn_options(options, *, da_cycling=False):
    opts = MynnOptions.from_namelist(options)
    if (opts.tkeadvect or opts.tkebudget
            or (opts.mixlength, opts.closure) not in ((2, 2.5), (1, 2.6))
            or (opts.mixlength == 1 and opts.edmf_mom != 1)
            or opts.cloudpdf != 2 or opts.stfunc != 1
            or opts.topdown or opts.scaleaware != 1 or opts.edmf != 1
            or opts.edmf_dd or opts.edmf_output or opts.edmf_mom not in (0, 1)
            or opts.edmf_tke
            or opts.mixscalars != 1 or opts.mixclouds != 1 or opts.mixqt
            or opts.dheat_opt not in (0, 1) or da_cycling):
        raise ValueError('MYNN Metal slot requires the supported non-cycling MYNN options')
    return opts


class MetalMpasBlMynn:
    """Native MYNN source path; state resides in native pools.

    The caller owns the enclosing transaction and must roll it back if a
    failure occurs while publishing the completed private columns.
    """
    corrections=frozenset({'RP1','RP7'})

    def __init__(self, context):self.context=context
    def restart_state(self):return {}
    def restore_restart_state(self,payload):
        if payload:raise ValueError('MYNN physical state belongs to native pools')

    def launch_slot(self,call):
        if (call.slot,call.scheme)!=('pbl','bl_mynn'):raise ValueError('MYNN slot requires native bl_mynn')
        p=call.pools;cfg=p['configs'];scalar=p['scalar_state']
        opts=validate_mynn_options(call.options,da_cycling=cfg['config_do_DAcycling'])
        nc,nl=int(p['nCellsSolve']),int(p['nVertLevels']);ctx=self.context;xp=_arrays(ctx)
        if nc<1 or nl<3:raise ValueError('MYNN slot needs columns and at least three levels')
        prep=p['prepared'];diag=p['diag_physics'];tend=p['tend_physics'];surf=p['sfc_input'];mesh=p['mesh'];raw=p['constants']
        flags={n:bool(scalar[n]) for n in ('f_qc','f_qi','f_qs','f_nc','f_ni','f_nifa','f_nwfa','f_nbca','f_qoz')}
        if flags['f_nbca'] or flags['f_qoz'] or not (flags['f_nc']==flags['f_nifa']==flags['f_nwfa']):
            raise ValueError('MYNN Metal slot requires absent black-carbon/ozone and jointly enabled aerosol-number species')
        def trans(src,*,reverse=False,dst=None):
            if dst is None:dst=xp.empty((nl,nc) if reverse else (nc,nl),np.float32)
            _require('MYNN source',src,np.float32,(nc,nl) if reverse else (nl,nc));_require('MYNN target',dst,np.float32,(nl,nc) if reverse else (nc,nl))
            if _context(src) is not ctx or _context(dst) is not ctx:raise ValueError('MYNN arrays belong to another context')
            _launch('hexv842_sw_layout_f32',nc*nl,ctx,dict(src=src,dst=dst),dict(nc=np.int32(nc),nl=np.int32(nl),reverse=np.int32(reverse),divide=np.int32(0),divisor=np.float32(1)))
            return dst
        def profile(name):return trans(prep[name][:nl])
        f={key:profile(name) for key,name in (('dz1','dz_p'),('u1','u_p'),('v1','v_p'),('w1','w_p'),('th1','th_p'),('tk1','t_p'),('p1','pres_hyd_p'),('ex1','pi_p'),('rho1','rho_p'))}
        mixing={'qv':profile('qv_p')}
        for name,flag in (('qc','f_qc'),('qi','f_qi'),('qs','f_qs')):mixing[name]=profile(name+'_p') if flags[flag] else xp.zeros((nc,nl),np.float32)
        spec={n:xp.zeros((nc,nl),np.float32) for n in ('sqv','sqc','sqi','sqs')}
        ops.pre(mixing,spec,(flags['f_qc'],flags['f_qi'],flags['f_qs']),exact_inverse='RP1' in getattr(call,'corrections',()))
        f.update({n+'1':v for n,v in spec.items()})
        for name,flag in (('qnc1','f_nc'),('qni1','f_ni'),('qnifa1','f_nifa'),('qnwfa1','f_nwfa')):
            native={'qnc1':'nc_p','qni1':'ni_p','qnifa1':'nifa_p','qnwfa1':'nwfa_p'}[name]
            f[name]=profile(native) if flags[flag] else xp.zeros((nc,nl),np.float32)
        f['qnbca1']=xp.zeros((nc,nl),np.float32);f['qozone1']=xp.zeros((nc,nl),np.float32)
        for name,native in (('el_pbl1','el_pbl'),('qke1','qke'),('cov1','cov'),('qsq1','qsq'),('tsq1','tsq'),('sh1','sh3d'),('sm1','sm3d'),('qc_bl1','qc_bl'),('qi_bl1','qi_bl'),('cldfra_bl1','cldfrac_bl')):f[name]=trans(diag[native][:nl])
        for name,arr in (('xland1',surf['xland']),('ts1',surf['skintemp']),('ust1',diag['ust']),('hfx1',diag['hfx']),('qfx1',diag['qfx']),('rmol_in',diag['rmol']),('wspd1',diag['wspd'])):f[name]=arr.copy()
        f['ps1']=prep['psfc_p'].copy();f['uoce1']=xp.zeros((nc,),np.float32);f['voce1']=xp.zeros((nc,),np.float32)
        f['dx1']=xp.empty((nc,),np.float32);ops.grid_spacing(mesh['meshDensity'],cfg['config_len_disp'],f['dx1'])
        for name in ('edmf_a1','edmf_w1'):f[name]=xp.zeros((nc,nl),np.float32)
        for name in ('qv1','qc1','qi1','qs1','sqw','thl','thetav'):f[name]=xp.empty((nc,nl),np.float32)
        f['zw']=xp.empty((nc,nl+1),np.float32);f['kpbl1']=xp.empty((nc,),np.int32);f['kpbl1'].fill(1)
        for name in ('pblh1','psig_bl','psig_shcu','flqv','flqc','th_sfc','flq','flt','fltv','rmol1','zet','pmz','phh'):f[name]=xp.empty((nc,),np.float32)
        c={n:np.float32(raw[n]) for n in ('cp','cpv','cice','cliq','ep_1','ep_2','gravity','karman','p0','r_d','r_v','svpt0','xlf','xls','xlv')}
        c.update(ep1=c['ep_1'],ep_3=np.float32(np.float32(1)-c['ep_2']),p1000mb=c['p0'],p608=c['ep_1'],t0c=c['svpt0'],rcp=np.float32(c['r_d']/c['cp']),rvovrd=np.float32(c['r_v']/c['r_d']),xlvcp=np.float32(c['xlv']/c['cp']),xlscp=np.float32(np.float32(c['xlv']+c['xlf'])/c['cp']))
        delt=np.float32(scalar['dt_pbl']);cold=bool(not cfg['config_do_restart'] and call.itimestep<=1)
        s = advance_plumes(f, constants=c, cold=cold,
                           momentum_opt=bool(opts.edmf_mom), tke_opt=bool(opts.edmf_tke),
                           scalar_opt=bool(opts.mixscalars), delt=delt,
                           mixlength=opts.mixlength, closure=opts.closure)
        tf=dict(flag_qc=flags['f_qc'],flag_qi=flags['f_qi'],flag_qs=flags['f_qs'],flag_qnc=flags['f_nc'],flag_qni=flags['f_ni'],flag_qnifa=flags['f_nifa'],flag_qnwfa=flags['f_nwfa'],flag_qnbca=flags['f_nbca'],flag_ozone=flags['f_qoz'],bl_mynn_cloudmix=bool(opts.mixclouds),bl_mynn_mixqt=bool(opts.mixqt),bl_mynn_edmf_mom=bool(opts.edmf_mom),bl_mynn_mixscalars=bool(opts.mixscalars))
        result=finish_columns(f,s,constants=c,flags=tf,delt=delt,dheatopt=bool(opts.dheat_opt),rp7='RP7' in getattr(call,'corrections',()));t=result['tendencies']
        ops.post(mixing,{n:t[n] for n in ('dqv','dqc','dqi','dqs')},(flags['f_qc'],flags['f_qi'],flags['f_qs']),delt,exact_inverse='RP1' in getattr(call,'corrections',()))
        # Publish only after all numerical stages complete. The enclosing
        # session transaction owns rollback if a publication operation fails.
        for native,source in (('el_pbl','el_pbl1'),('qke','qke1'),('cov','cov1'),('qsq','qsq1'),('tsq','tsq1'),('sh3d','sh1'),('sm3d','sm1'),('qc_bl','qc_bl1'),('qi_bl','qi_bl1'),('cldfrac_bl','cldfra_bl1')):trans(f[source],reverse=True,dst=diag[native][:nl])
        for native,source in (('rublten','du'),('rvblten','dv'),('rthblten','dth'),('rqvblten','dqv'),('rqcblten','dqc'),('rqiblten','dqi')):
            trans(t[source],reverse=True,dst=prep[native+'_p'][:nl]);trans(t[source],reverse=True,dst=tend[native][:nl])
        trans(t['dqs'],reverse=True,dst=tend['rqsblten'][:nl])
        if flags['f_ni']:trans(t['dqni'],reverse=True,dst=tend['rniblten'][:nl])
        if flags['f_nc'] and flags['f_nifa'] and flags['f_nwfa']:
            for native,source in (('rncblten','dqnc'),('rnifablten','dqnifa'),('rnwfablten','dqnwfa')):trans(t[source],reverse=True,dst=tend[native][:nl])
        for name,value in (('kzh',result['exch_h']),('kzm',result['exch_m'])):trans(value,reverse=True,dst=prep[name+'_p'][:nl])
        for name,value in (('kzh',result['exch_h']),('kzm',result['exch_m']),('kzq',result['exch_h'])):trans(value,reverse=True,dst=diag[name][:nl])
        zero=xp.zeros((nc,nl),np.float32)
        for name in ('dqke','qbuoy','qdiss','qshear','qwt','edmf_a','edmf_w','edmf_qt','edmf_thl','edmf_ent','edmf_qc','sub_thl','sub_qv','det_thl','det_qv'):trans(zero,reverse=True,dst=diag[name][:nl])
        ctx.submit_bookkeeping(((f['pblh1'],diag['hpbl']),(f['kpbl1'],diag['kpbl'])))
        return result
