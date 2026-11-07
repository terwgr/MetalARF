"""Whole selected-host MYNN oracle for one actual coupled SlotCall.

The snapshot is read only for comparison; production runs the Metal slot.
The existing host shims do not imply NVIDIA execution.
"""
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_bl_mynn.probe_whole_host import host_module
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.native_powf import CUDA_NATIVE_POWF
from woof.hex.mpasphys.bl_mynn_v842 import (ADAPTER_CUDA_SOURCE,MYNN_IEEE_ARITHMETIC_SUPPORT,
    MYNN_ARGUMENTS,MYNN_WORKSPACE_LEVEL_BYTES,MYNN_WORKSPACE_FIXED_BYTES)


class MynnOracle:
    def __init__(self):
        a=ADAPTER_CUDA_SOURCE.index('extern "C" __global__ void mynn_pre(')
        b=ADAPTER_CUDA_SOURCE.index('// MPAS mpas_atmphys_driver_pbl.F:454-485',a)
        source='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+ADAPTER_CUDA_SOURCE[a:b]
        self.adapter=HostModule('coupled_mynn_adapter',source)
        a=ADAPTER_CUDA_SOURCE.index('extern "C" __global__ void mynn_dx(')
        source='#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+CUDA_NATIVE_POWF+MYNN_IEEE_ARITHMETIC_SUPPORT+ADAPTER_CUDA_SOURCE[a:]
        self.dx=HostModule('coupled_mynn_dx',source).get_function('mynn_dx')
        self.host=host_module();self.calls=0;self.words=0

    def compare(self,call,launch):
        if {'RP1','RP7'} & set(call.corrections):
            raise ValueError('coupled MYNN oracle covers the native RP1/RP7 path')
        p=call.pools;nc,nl=int(p['nCellsSolve']),int(p['nVertLevels']);n=nc*nl
        prep,diag,tend,surf,mesh=(p[k] for k in ('prepared','diag_physics','tend_physics','sfc_input','mesh'))
        flags={name:bool(p['scalar_state'][name]) for name in ('f_qc','f_qi','f_qs','f_nc','f_ni','f_nifa','f_nwfa')}
        if bool(p['scalar_state']['f_nbca']) or bool(p['scalar_state']['f_qoz']):
            raise ValueError('coupled MYNN oracle excludes black carbon and ozone')
        cold=not p['configs']['config_do_restart'] and call.itimestep<=1
        dt=np.float32(p['scalar_state']['dt_pbl'])
        raw=p['constants'];constant_names=('cp','cpv','cice','cliq','ep_1','ep_2','gravity','karman','p0','r_d','r_v','svp1','svp2','svp3','svpt0','xlf','xls','xlv')
        cvals=np.array([raw[name] for name in constant_names],np.float32)
        zero=lambda:np.zeros((nc,nl),np.float32,order='F')
        def row(value):return np.asfortranarray(value.to_numpy()[:nl].T)
        v={}
        for name,specification in MYNN_ARGUMENTS.items():
            if name in ('errmsg','errflg'):continue
            if specification['dimensions']:
                dtype=np.int32 if specification['kind']=='int' else np.float32
                shape=(nc,) if len(specification['dimensions'])==1 else (nc,nl+1 if name=='w' else nl)
                v[name]=np.zeros(shape,dtype,order='F')
            else:v[name]=np.float32(0) if specification['kind']=='float' else np.int32(0)
        for name,native_name in (('dz','dz_p'),('u','u_p'),('v','v_p'),('th','th_p'),('tt','t_p'),('p','pres_hyd_p'),('exner','pi_p'),('rho','rho_p')):
            v[name]=row(prep[native_name])
        for name,native_name,flag in (('qnc','nc_p','f_nc'),('qni','ni_p','f_ni'),('qnifa','nifa_p','f_nifa'),('qnwfa','nwfa_p','f_nwfa')):
            v[name]=row(prep[native_name]) if flags[flag] else zero()
        v['w'][:,:]=np.asfortranarray(prep['w_p'].to_numpy()[:nl+1].T)
        self.dx(((nc+127)//128,),(128,),(nc,np.float32(p['configs']['config_len_disp']),mesh['meshDensity'].to_numpy(),v['dx']))
        v['xland']=surf['xland'].to_numpy();v['ts']=surf['skintemp'].to_numpy();v['ps']=prep['psfc_p'].to_numpy()
        for name in ('ust','hfx','qfx','rmol','wspd'):v[name]=diag[name].to_numpy()
        for name,native_name in (('el_pbl','el_pbl'),('qke','qke'),('cov','cov'),('qsq','qsq'),('tsq','tsq'),('sh','sh3d'),('sm','sm3d'),('qc_bl','qc_bl'),('qi_bl','qi_bl'),('cldfra_bl','cldfrac_bl'),('edmf_a','edmf_a'),('edmf_w','edmf_w'),('edmf_qt','edmf_qt'),('edmf_thl','edmf_thl'),('edmf_ent','edmf_ent'),('edmf_qc','edmf_qc'),('sub_thl','sub_thl'),('sub_sqv','sub_qv'),('det_thl','det_thl'),('det_sqv','det_qv')):
            v[name]=row(diag[native_name])
        mix={'qv':row(prep['qv_p'])}
        for name in ('qc','qi','qs'):mix[name]=row(prep[name+'_p']) if flags['f_'+name] else zero()
        spec={name:zero() for name in ('sqv','sqc','sqi','sqs')}
        enabled=tuple(int(flags['f_'+name]) for name in ('qc','qi','qs'))
        self.adapter.get_function('mynn_pre')(((n+127)//128,),(128,),(n,*enabled,*mix.values(),*spec.values()));v.update(spec)
        v.update(initflag=int(cold),restart=int(p['configs']['config_do_restart']),cycling=0,delt=dt,icloud_bl=0,
            bl_mynn_tkeadvect=0,bl_mynn_tkebudget=0,bl_mynn_cloudpdf=2,bl_mynn_mixlength=2,bl_mynn_closure=np.float32(2.5),
            bl_mynn_stfunc=1,bl_mynn_topdown=0,bl_mynn_edmf=1,bl_mynn_edmf_dd=0,bl_mynn_edmf_mom=0,bl_mynn_edmf_tke=0,
            bl_mynn_mixscalars=1,bl_mynn_output=0,bl_mynn_cloudmix=1,bl_mynn_mixqt=0,bl_mynn_scaleaware=1,bl_mynn_dheatopt=1,
            spp_pbl=0,flag_qc=int(flags['f_qc']),flag_qi=int(flags['f_qi']),flag_qs=int(flags['f_qs']),
            flag_qnc=int(flags['f_nc']),flag_qni=int(flags['f_ni']),flag_qnwfa=int(flags['f_nwfa']),flag_qnifa=int(flags['f_nifa']),
            flag_qnbca=0,flag_qoz=0,its=1,ite=nc,kts=1,kte=nl,kme=nl+1)
        v['kpbl'].fill(1)
        stride=(MYNN_WORKSPACE_LEVEL_BYTES*nl+MYNN_WORKSPACE_FIXED_BYTES+7)//8*8
        bound=((nc+31)//32)*32*stride;workspace=np.full(bound+16384,0xa5,np.uint8);errors=np.zeros(nc,np.int32)
        args=[name for name in MYNN_ARGUMENTS if name not in ('errmsg','errflg')]
        self.host((1,),(128,),(0,nc,cvals,workspace,np.uint64(stride),errors,*[v[name] for name in args]))
        if np.any(errors) or np.any(workspace[bound:]!=0xa5):raise AssertionError('coupled MYNN source error or workspace overrun')
        self.adapter.get_function('mynn_post')(((n+127)//128,),(128,),
            (n,*enabled,dt,*mix.values(),*[v[name] for name in ('rqvblten','rqcblten','rqiblten','rqsblten')]))
        comparisons=[]
        for name,ref in (('el_pbl','el_pbl'),('qke','qke'),('cov','cov'),('qsq','qsq'),('tsq','tsq'),('sh3d','sh'),('sm3d','sm'),('qc_bl','qc_bl'),('qi_bl','qi_bl'),('cldfrac_bl','cldfra_bl'),('edmf_a','edmf_a'),('edmf_w','edmf_w'),('edmf_qt','edmf_qt'),('edmf_thl','edmf_thl'),('edmf_ent','edmf_ent'),('edmf_qc','edmf_qc'),('sub_thl','sub_thl'),('sub_qv','sub_sqv'),('det_thl','det_thl'),('det_qv','det_sqv'),('dqke','dqke'),('qbuoy','qbuoy'),('qdiss','qdiss'),('qshear','qshear'),('qwt','qwt'),('kzh','exch_h'),('kzm','exch_m'),('kzq','exch_h')):
            comparisons.append((diag,name,v[ref].T))
        comparisons.extend(((diag,'hpbl',v['pblh']),(diag,'kpbl',v['kpbl'])))
        for name,ref,flag in (('rublten','rublten',True),('rvblten','rvblten',True),('rthblten','rthblten',True),('rqvblten','rqvblten',True),('rqcblten','rqcblten',True),('rqiblten','rqiblten',True),('rqsblten','rqsblten',True),('rniblten','rqniblten',flags['f_ni']),('rncblten','rqncblten',flags['f_nc']),('rnifablten','rqnifablten',flags['f_nifa']),('rnwfablten','rqnwfablten',flags['f_nwfa'])):
            comparisons.append((tend,name,v[ref].T if flag else tend[name].to_numpy()))
        for name,ref in (('rublten_p','rublten'),('rvblten_p','rvblten'),('rthblten_p','rthblten'),('rqvblten_p','rqvblten'),('rqcblten_p','rqcblten'),('rqiblten_p','rqiblten'),('kzh_p','exch_h'),('kzm_p','exch_m')):
            comparisons.append((prep,name,v[ref].T))
        launch(call)
        for pool,name,want in comparisons:
            got=pool[name].to_numpy()[:nl] if want.ndim==2 else pool[name].to_numpy()
            if got.dtype!=want.dtype or got.shape!=want.shape or got.tobytes()!=np.ascontiguousarray(want).tobytes():
                raise AssertionError('coupled MYNN selected-source mismatch: '+name)
            self.words+=got.size
        self.calls+=1
        return len(comparisons)
