"""Compare a bounded MYNN SlotCall with the unpartitioned selected host module.

Host-only shims replace CUDA PTX selection/negation and resolve a sequencing
ambiguity in the descriptor implementation; no NVIDIA execution is implied.
"""
from types import SimpleNamespace
import numpy as np
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_bl_mynn.probe_whole_host import host_module
from tools.mpasphys_v842_bl_mynn.exercise_slot import host_pools
from woof.hex.cuda_fp32 import CUDA_FTZ_HELPERS
from woof.hex.mpasphys.bl_mynn_v842 import (ADAPTER_CUDA_SOURCE,MYNN_IEEE_ARITHMETIC_SUPPORT,MYNN_ARGUMENTS,MYNN_WORKSPACE_LEVEL_BYTES,MYNN_WORKSPACE_FIXED_BYTES)
from woof.hex.mpasphys.metal_bl_mynn_slot_v842 import MetalMpasBlMynn
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule


def main():
    source=host_pools();nc,nl=source['nCellsSolve'],source['nVertLevels'];raw=source['constants'];raw.update(xlv=np.float32(2490*1004),xlf=np.float32((2819-2490)*1004),svp1=np.float32(0),svp2=np.float32(0),svp3=np.float32(0))
    source['prepared']['psfc_p'][:2]=np.array([99062.5,99063.0],np.float32)
    a=ADAPTER_CUDA_SOURCE.index('extern "C" __global__ void mynn_pre(');b=ADAPTER_CUDA_SOURCE.index('// MPAS mpas_atmphys_driver_pbl.F:454-485',a)
    mod=HostModule('mpas_v842_mynn_whole_adapter','#define __CUDA_ARCH__ 1\n'+CUDA_FTZ_HELPERS+MYNN_IEEE_ARITHMETIC_SUPPORT+ADAPTER_CUDA_SOURCE[a:b]);hosts={False:host_module(),True:host_module(rp7=True)}
    prep=source['prepared'];diag=source['diag_physics'];zero=lambda:np.zeros((nc,nl),np.float32,order='F')
    def row(x):return np.asfortranarray(x.T)
    cvals=np.array([raw[n] for n in ('cp','cpv','cice','cliq','ep_1','ep_2','gravity','karman','p0','r_d','r_v','svp1','svp2','svp3','svpt0','xlf','xls','xlv')],np.float32)
    with MetalContext(batch_kernel_calls=False) as ctx:
        submit=ctx._submit_kernel_batch
        def sync(*args,**kwargs):
            result=submit(*args,**kwargs);result.wait();return result
        ctx._submit_kernel_batch=sync;xp=MetalArrayModule(ctx)
        snapshots={}
        for cold,inverse,absent,rp7 in ((True,False,False,False),(False,False,False,False),(True,True,False,False),(False,True,False,False),(False,False,True,False),(True,False,False,True),(False,False,False,True),(True,True,False,True),(False,True,False,True),(False,False,True,True),(False,True,True,True)):
            pools={k:({a:xp.asarray(b.copy()) for a,b in v.items()} if k in ('prepared','diag_physics','tend_physics','sfc_input','mesh') else dict(v) if isinstance(v,dict) else v) for k,v in source.items()}
            if absent:
                for name in ('f_qc','f_qi','f_qs','f_nc','f_ni','f_nifa','f_nwfa'):pools['scalar_state'][name]=0
            actual=SimpleNamespace(slot='pbl',scheme='bl_mynn',pools=pools,options={},corrections=frozenset(({'RP1'} if inverse else set()) | ({'RP7'} if rp7 else set())),itimestep=1 if cold else 2)
            MetalMpasBlMynn(ctx).launch_slot(actual)
            v={}
            for name,spec in MYNN_ARGUMENTS.items():
                if name in ('errmsg','errflg'):continue
                if spec['dimensions']:
                    dtype=np.int32 if spec['kind']=='int' else np.float32
                    shape=(nc,) if len(spec['dimensions'])==1 else (nc,nl+1 if name=='w' else nl)
                    v[name]=np.zeros(shape,dtype,order='F')
                else:v[name]=np.float32(0) if spec['kind']=='float' else np.int32(0)
            for name,native in (('dz','dz_p'),('u','u_p'),('v','v_p'),('th','th_p'),('tt','t_p'),('p','pres_hyd_p'),('exner','pi_p'),('rho','rho_p'),('qnc','nc_p'),('qni','ni_p'),('qnifa','nifa_p'),('qnwfa','nwfa_p')):v[name]=row(prep[native][:nl])
            v['w'][:,:]=row(prep['w_p'][:nl+1]);v['dx'].fill(3000);v['xland']=source['sfc_input']['xland'].copy();v['ts']=source['sfc_input']['skintemp'].copy();v['ps']=prep['psfc_p'].copy()
            for name in ('ust','hfx','qfx','rmol','wspd'):v[name]=diag[name].copy()
            for name,native in (('el_pbl','el_pbl'),('qke','qke'),('cov','cov'),('qsq','qsq'),('tsq','tsq'),('sh','sh3d'),('sm','sm3d'),('qc_bl','qc_bl'),('qi_bl','qi_bl'),('cldfra_bl','cldfrac_bl'),('edmf_a','edmf_a'),('edmf_w','edmf_w'),('edmf_qt','edmf_qt'),('edmf_thl','edmf_thl'),('edmf_ent','edmf_ent'),('edmf_qc','edmf_qc'),('sub_thl','sub_thl'),('sub_sqv','sub_qv'),('det_thl','det_thl'),('det_sqv','det_qv')):v[name]=row(diag[native])
            mix={n:(zero() if absent and n!='qv' else row(prep[n+'_p'])) for n in ('qv','qc','qi','qs')};spec={n:zero() for n in ('sqv','sqc','sqi','sqs')}
            if absent:
                for n in ('qnc','qni','qnifa','qnwfa'):v[n]=zero()
            pre=mod.get_function('mynn_pre_rp1' if inverse else 'mynn_pre');present=int(not absent)
            pre(((nc*nl+127)//128,),(128,),(nc*nl,present,present,present,*mix.values(),*spec.values()));v.update(spec)
            v.update(initflag=int(cold),restart=0,cycling=0,delt=np.float32(30),icloud_bl=0,bl_mynn_tkeadvect=0,bl_mynn_tkebudget=0,bl_mynn_cloudpdf=2,bl_mynn_mixlength=2,bl_mynn_closure=np.float32(2.5),bl_mynn_stfunc=1,bl_mynn_topdown=0,bl_mynn_edmf=1,bl_mynn_edmf_dd=0,bl_mynn_edmf_mom=0,bl_mynn_edmf_tke=0,bl_mynn_mixscalars=1,bl_mynn_output=0,bl_mynn_cloudmix=1,bl_mynn_mixqt=0,bl_mynn_scaleaware=1,bl_mynn_dheatopt=1,spp_pbl=0,flag_qc=1,flag_qi=1,flag_qs=1,flag_qnc=1,flag_qni=1,flag_qnwfa=1,flag_qnifa=1,flag_qnbca=0,flag_qoz=0,its=1,ite=nc,kts=1,kte=nl,kme=nl+1)
            if absent:
                for name in ('flag_qc','flag_qi','flag_qs','flag_qnc','flag_qni','flag_qnwfa','flag_qnifa'):v[name]=0
            v['kpbl'].fill(1)
            guards={}
            for name,value in tuple(v.items()):
                if isinstance(value,np.ndarray):
                    base=np.full(value.size+16384,0x7f123456,np.uint32)
                    view=base[:value.size].view(value.dtype).reshape(value.shape,order='F');view[...] = value
                    guards[name]=base;v[name]=view
            stride=(MYNN_WORKSPACE_LEVEL_BYTES*nl+MYNN_WORKSPACE_FIXED_BYTES+7)//8*8;warp_count=(nc+31)//32;bound=warp_count*32*stride;workspace=np.full(bound+16384,0xa5,np.uint8);errors=np.zeros(nc,np.int32)
            names=[name for name in MYNN_ARGUMENTS if name not in ('errmsg','errflg')]
            hosts[rp7]((1,),(128,),(0,nc,cvals,workspace,np.uint64(stride),errors,*[v[n] for n in names]))
            if np.any(errors):raise AssertionError('source returned error')
            if np.any(workspace[bound:]!=0xa5):raise AssertionError('source exceeded warp workspace')
            for name,base in guards.items():
                if np.any(base[v[name].size:]!=0x7f123456):raise AssertionError('source wrote outside '+name)
            post=mod.get_function('mynn_post_rp1' if inverse else 'mynn_post')
            post(((nc*nl+127)//128,),(128,),(nc*nl,present,present,present,np.float32(30),*mix.values(),*[v[n] for n in ('rqvblten','rqcblten','rqiblten','rqsblten')]))
            for name,base in guards.items():
                if np.any(base[v[name].size:]!=0x7f123456):raise AssertionError('post wrote outside '+name)
            comparisons=[]
            for name,ref in (('el_pbl','el_pbl'),('qke','qke'),('cov','cov'),('qsq','qsq'),('tsq','tsq'),('sh3d','sh'),('sm3d','sm'),('qc_bl','qc_bl'),('qi_bl','qi_bl'),('cldfrac_bl','cldfra_bl'),('edmf_a','edmf_a'),('edmf_w','edmf_w'),('edmf_qt','edmf_qt'),('edmf_thl','edmf_thl'),('edmf_ent','edmf_ent'),('edmf_qc','edmf_qc'),('sub_thl','sub_thl'),('sub_qv','sub_sqv'),('det_thl','det_thl'),('det_qv','det_sqv'),('dqke','dqke'),('qbuoy','qbuoy'),('qdiss','qdiss'),('qshear','qshear'),('qwt','qwt'),('kzh','exch_h'),('kzm','exch_m'),('kzq','exch_h')):comparisons.append(('diag_physics',name,v[ref].T))
            comparisons.extend((('diag_physics','hpbl',v['pblh']),('diag_physics','kpbl',v['kpbl'])))
            for name,ref in (('rublten','rublten'),('rvblten','rvblten'),('rthblten','rthblten'),('rqvblten','rqvblten'),('rqcblten','rqcblten'),('rqiblten','rqiblten'),('rqsblten','rqsblten'),('rniblten','rqniblten'),('rncblten','rqncblten'),('rnifablten','rqnifablten'),('rnwfablten','rqnwfablten')):comparisons.append(('tend_physics',name,v[ref].T))
            for name,ref in (('rublten_p','rublten'),('rvblten_p','rvblten'),('rthblten_p','rthblten'),('rqvblten_p','rqvblten'),('rqcblten_p','rqcblten'),('rqiblten_p','rqiblten'),('kzh_p','exch_h'),('kzm_p','exch_m')):comparisons.append(('prepared',name,v[ref].T))
            for pool,name,want in comparisons:
                got=pools[pool][name].to_numpy();bad=np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
                if bad.size:raise AssertionError(f'cold={cold} {pool}.{name}: {bad.size} differences at {bad[0]} got={got.flat[bad[0]]} expected={want.flat[bad[0]]}')
            print(f'MYNN full selected-source host comparison cold={cold} RP1={inverse} RP7={rp7} absent={absent} matches {len(comparisons)} output fields',flush=True)
            if not absent:snapshots[cold,inverse,rp7]=pools['tend_physics']['rqvblten'].to_numpy().copy()
        for cold,inverse in ((True,False),(False,False),(True,True),(False,True)):
            exact=snapshots[cold,inverse,False];corrected=snapshots[cold,inverse,True]
            if np.array_equal(exact.view(np.uint32),corrected.view(np.uint32)):raise AssertionError(f'RP7 did not change output: cold={cold} RP1={inverse} exact={exact[:3,:2]} corrected={corrected[:3,:2]}')
            if np.isfinite(exact[0,0]) or not np.isfinite(corrected[0,0]):raise AssertionError('zero pressure-difference case did not discriminate RP7')
        print('RP7 corrects the zero-pressure-difference vapor tendency',flush=True)
if __name__=='__main__':main()
