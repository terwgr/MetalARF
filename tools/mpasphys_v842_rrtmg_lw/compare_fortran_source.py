"""Compare fixed-depth native LW stage composition using authenticated tables."""
from pathlib import Path
import re,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.thompson_real_column_parity.host_backend import HostModule
from tools.mpasphys_v842_rrtmg_lw.band_source import band_source
from tools.mpasphys_v842_rrtmg_lw.compare_upper import host_kernel as upper_host
from woof.core.rrtmg_lw import _lw_coalesced_source, GPU_BAND_TABS
from woof.hex.mpasphys.rrtmg_lw_v842 import _solver_source,_DRIVER_SOURCE,_RANDOM_SOURCE,GPU_FSLOTS,GPU_ISLOTS
from woof.hex.mpasphys.metal_rrtmg_lw_v842 import load_tables,solve_group,run_group,_SETCOEF_FLOAT_OUTPUTS,_SETCOEF_INT_OUTPUTS
from woof.metal.runtime import MetalContext
from woof.metal.array import MetalArrayModule
solver=_solver_source();chain=_lw_coalesced_source((ROOT/'woof/core/kernels/rrtmg_lw_chain.cu').read_text())
def extract(source,name):
    m=re.search(r'(?:extern "C" __global__ void|__device__ (?:float|void)) '+name+r'\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    return source[m.start():j]
header=(ROOT/'woof/core/kernels/glibc_flt32.cuh').read_text()+'''
#define NGPTLW 140
#define NBNDLW 16
#define MAXXSEC 4
#define MXMOL 38
#define RLW_MAXLAY 128
#define RLW_REC6 0.166667f
#define RLW_TBLINT 10000.0f
#define RLW_AD(a,b) __fadd_rn(a,b)
#define RLW_SU(a,b) __fsub_rn(a,b)
#define RLW_MU(a,b) __fmul_rn(a,b)
#define RLW_DV(a,b) __fdiv_rn(a,b)
#define rlw_log(x) gfk_log(x)
#define rlw_exp(x) gfk_exp(x)
#define rlw_pow(x,y) gfk_pow(x,y)
__device__ float rlw_pow4(float x){float a=RLW_MU(x,x);return RLW_MU(a,a);}
'''
macros=_DRIVER_SOURCE[_DRIVER_SOURCE.index('#define M(f,k)'):_DRIVER_SOURCE.index('// MPAS: module_ra_rrtmg_lw.F:11954-11983.')]
chosen='\n'.join(extract(chain,n) for n in ('rlw_inatm_coalesced','rlw_cldprmc_coalesced','rlw_taut_coalesced','rlw_rtrn_secdiff_coalesced','rlw_rtrn_prol_coalesced','rlw_rtrn_march_coalesced','rlw_rtrn_accum_coalesced','rlw_rtrn_final_coalesced'))
chosen+=re.sub(r'\btz\b','tlevel',extract(solver,'rlw_setcoef'))
constants='\n'.join(re.search(r'__device__ const float mpas_'+n+r'\[\d+\] = \{.*?\};',_DRIVER_SOURCE,re.S).group() for n in ('retab','pprof','tprof','o3sum','ppsum','o3win','ppwin'))
chosen+=_RANDOM_SOURCE+constants+'\n'+macros+'\n'+'\n'.join(extract(_DRIVER_SOURCE,n) for n in ('mpas_profile','mpas_ice_radius','mpas_liquid_radius','mpas_vinterp_one','mpas_lw_prepare_one','mpas_lw_prepare','mpas_kiss','mpas_mcica_make','mpas_lw_mcica','mpas_lw_outputs_one','mpas_lw_outputs'))
host=HostModule('mpas_v842_lw_solver',header+chosen)
def call(name,count,args):host.get_function(name)(((count+127)//128,),(128,),args)
def lower_kernel(number):
    if number>=9:return upper_host(number)
    source=solver;m=re.search(r'extern "C" __global__ void rlw_taugb'+str(number)+r'\s*\(',source);i=source.index('{',m.end());j=i+1;depth=1
    while depth:
        if source[j]=='{':depth+=1
        elif source[j]=='}':depth-=1
        j+=1
    _,_,names=band_source(number);kernel=source[m.start():j]
    kernel=kernel.replace('const float* const* __restrict__ tabs,',','.join('const float* __restrict__ in_'+n for n in names)+',')
    for ix,n in enumerate(names):kernel=re.sub(r'const float\s*\*\s*'+n+r'\s*=\s*tabs\['+str(ix)+r'\];',f'const float* {n} = in_{n};',kernel)
    common=source[source.index('#define SL_PAVEL 0'):source.index('extern "C" __global__ void rlw_taugb1(')]
    helpers=''
    if 3<=number<=5:
        a=(ROOT/'woof/core/kernels/rrtmg_lw_taugb03_05.cu').read_text();helpers=a[a.index('#define CHI35'):a.index('// ---------------------------------------------------------------------\n// Band 3:')]
    if 6<=number<=8:
        a=(ROOT/'woof/core/kernels/rrtmg_lw_taugb06_09.cu').read_text();helpers=a[a.index('#define KA3_0609'):a.index('// ---------------------------------------------------------------------\n// Band 6:')]
    return HostModule('mpas_v842_lw_solver_band'+str(number),header+common+helpers+kernel).get_function('rlw_taugb'+str(number))
from tools.rrtmg_wrf461_oracle.lw_fixtures import read_fixture
fixture=read_fixture(sys.argv[1])
get=lambda name:fixture['002/'+name]
nc,nm,nl=1,55,58
model=np.ascontiguousarray(get('model')[None]);interfaces=np.ascontiguousarray(get('interfaces')[None]);surface=np.ascontiguousarray(get('surface')[None]);pin=np.ascontiguousarray(get('pin'));ozone=np.ascontiguousarray(get('ozone')[None])
a=np.empty((nc,12,nl),np.float32);lev=np.empty((nc,2,nl+1),np.float32)
call('mpas_lw_prepare',nc,(*map(np.int32,(nc,nm,nl,1,0,0,0,0,7,15)),model,interfaces,surface,pin,ozone,a,lev))
with MetalContext(batch_kernel_calls=False) as context:
    submit=context._submit_kernel_batch
    def sync(*args,**kwargs):
        result=submit(*args,**kwargs);result.wait();return result
    context._submit_kernel_batch=sync;xp=MetalArrayModule(context)
    device_tables=load_tables(context)
    tables={n:(v.to_numpy() if hasattr(v,'to_numpy') else v) for n,v in device_tables.items() if n not in ('band','reduced')}
    tabs={b:{name:device_tables['reduced'][f'kg{b:02d}/{name}'].to_numpy() for name in GPU_BAND_TABS[b]} for b in range(1,17)}
    fs=np.zeros((33,nc,nl),np.float32);isv=np.zeros((6,nc,nl),np.int32);fs[0]=a[:,0,:]
    pz=lev[:,0,:].copy();tz=lev[:,1,:].copy();tavel=a[:,1,:].copy();tsfc=surface[:,0].copy();emiss=np.repeat(surface[:,1,None],16,axis=1).copy()
    wkl=np.zeros((nc,38,nl),np.float32);wx=np.zeros((nc,4,nl),np.float32)
    wkl[:,0,:]=a[:,3,:];wkl[:,1,:]=np.float32(379e-6);wkl[:,2,:]=a[:,4,:];wkl[:,3,:]=np.float32(319e-9);wkl[:,5,:]=np.float32(1774e-9);wkl[:,6,:]=np.float32(.209488)
    wx[:,0,:]=np.float32(.093e-9);wx[:,1,:]=np.float32(.251e-9);wx[:,2,:]=np.float32(.538e-9);wx[:,3,:]=np.float32(.169e-9)
    wbrodl=np.zeros((nc,nl),np.float32);pwv=np.zeros(nc,np.float32)
    call('rlw_inatm_coalesced',nc,(*map(np.int32,(nc,nl)),pz,tables['grav'],tables['avogad'],wkl,wx,fs[1],wbrodl,pwv))
    cloud=[np.zeros((nc,nl,140),np.float32) for _ in range(4)]
    call('mpas_lw_mcica',nc,(*map(np.int32,(nc,nl,2,0,277)),surface,a,*cloud))
    tau=np.zeros_like(cloud[0]);ncb=np.zeros(nc,np.int32);err=np.zeros(1,np.int32)
    call('rlw_cldprmc_coalesced',nc*140,(*map(np.int32,(nc,nl,2,3,1)),cloud[0],cloud[2],cloud[1],cloud[3],a[:,10,:].copy(),a[:,9,:].copy(),a[:,11,:].copy(),*[tables[n] for n in ('absice1','absice2','absice3','absice0','absliq1')],tables['absliq0'],tables['ngb'],tau,ncb,err))
    if err[0]:raise ValueError(f'source cloud optics error {err[0]}')
    laytrop=np.zeros(nc,np.int32);pl=np.zeros((nc,nl,16),np.float32);pv=np.zeros((nc,nl+1,16),np.float32);pb=np.zeros((nc,16),np.float32)
    slot={n:fs[GPU_FSLOTS.index(n)] for n in _SETCOEF_FLOAT_OUTPUTS};slot.update({n:isv[GPU_ISLOTS.index(n)] for n in _SETCOEF_INT_OUTPUTS})
    order=('jp','jt','jt1');rest=('colh2o','colco2','colo3','coln2o','colco','colch4','colo2','colbrd','fac00','fac01','fac10','fac11','rat_h2oco2','rat_h2oco2_1','rat_h2oo3','rat_h2oo3_1','rat_h2on2o','rat_h2on2o_1','rat_h2och4','rat_h2och4_1','rat_n2oco2','rat_n2oco2_1','rat_o3co2','rat_o3co2_1','selffac','selffrac','indself','forfac','forfrac','indfor','minorfrac','scaleminor','scaleminorn2','indminor')
    call('rlw_setcoef',nc,(*map(np.int32,(nc,nl,1)),fs[0],tavel,tz,tsfc,emiss,fs[1],wkl,wbrodl,*[tables[n] for n in ('totplnk','totplk16','preflog','tref','chi')],laytrop,*[slot[n] for n in order],pl,pv,pb,*[slot[n] for n in rest]))
    gas=np.zeros((nc,nl,140),np.float32);fracs=np.zeros_like(gas)
    for b in range(1,17):
        lower_kernel(b)(((nc*nl+127)//128,),(128,),(*map(np.int32,(nc,nl)),laytrop,fs,isv,wx,tables['chi'],tables['oneminus'],*[tabs[b][n] for n in GPU_BAND_TABS[b]],gas,fracs))
    total=np.zeros_like(gas);aerosol=np.zeros((nc,nl,16),np.float32)
    call('rlw_taut_coalesced',nc*nl*140,(*map(np.int32,(nc,nl)),gas,aerosol,tables['ngb'],total))
    sec=np.zeros((nc,16),np.float32);call('rlw_rtrn_secdiff_coalesced',nc,(np.int32(nc),pwv,*[tables[n] for n in ('a0','a1','a2')],sec))
    od=np.zeros_like(gas);ef=np.zeros_like(gas);flags=np.zeros((nc,nl),np.int32)
    call('rlw_rtrn_prol_coalesced',nc*nl,(*map(np.int32,(nc,nl)),cloud[0],tau,sec,tables['ngb'],od,ef,flags))
    march={n:np.zeros_like(gas) for n in ('radld_p','radclrd_p','radlu_p','radclru_p')};march['iclddn_p']=np.zeros(gas.shape,np.uint8)
    march.update({n:np.zeros((nc,140),np.float32) for n in ('radlu_sfc','radclru_sfc')})
    call('rlw_rtrn_march_coalesced',nc*140,(*map(np.int32,(nc,nl)),cloud[0],od,ef,flags,sec,emiss,pl,pv,pb,fracs,total,*[tables[n] for n in ('tau_tbl','exp_tbl','tfn_tbl')],tables['bpade'],tables['ngb'],*[march[n] for n in ('radld_p','radclrd_p','radlu_p','radclru_p','iclddn_p','radlu_sfc','radclru_sfc')]))
    result={n:np.zeros((nc,nl+1),np.float32) for n in ('totuflux','totdflux','totuclfl','totdclfl','fnet','fnetc','htr','htrc')}
    call('rlw_rtrn_accum_coalesced',nc*(nl+1),(*map(np.int32,(nc,nl)),*[march[n] for n in ('radld_p','radclrd_p','radlu_p','radclru_p','iclddn_p','radlu_sfc','radclru_sfc')],tables['ngs'],tables['delwave'],np.float32(.5),*[result[n] for n in ('totuflux','totdflux','totuclfl','totdclfl')]))
    call('rlw_rtrn_final_coalesced',nc,(*map(np.int32,(nc,nl)),pz,tables['fluxfac'],tables['heatfac'],*[result[n] for n in ('totuflux','totdflux','totuclfl','totdclfl','fnet','fnetc','htr','htrc')]))
    assert np.all(np.isfinite(result['htr'])) and np.any(result['htr']!=0) and np.any(result['totuflux']!=0), 'native transfer must be active'
    tendency=np.empty((nc,nm),np.float32);radii=np.empty((nc,3,nm),np.float32);flux=np.empty((nc,11),np.float32)
    call('mpas_lw_outputs',nc,(*map(np.int32,(nc,nm,nl,1)),model,a,result['totuflux'],result['totdflux'],result['totuclfl'],result['totdclfl'],result['htr'][:,:nl].copy(),tendency,radii,flux))
    dm=xp.asarray(model);di=xp.asarray(interfaces);ds=xp.asarray(surface);dp=xp.asarray(pin);do=xp.asarray(ozone)
    got=run_group(dm,di,ds,dp,do,device_tables,icloud=1,overlap=2,idcor=0,julday=277,reqc=0,reqi=0,reqs=0)
    count=bad=0
    for name,want in (('tendency',tendency),('radii',radii),('flux',flux)):
        actual=got[name].to_numpy();diff=np.argwhere(actual.view(np.uint32)!=want.view(np.uint32));count+=want.size;bad+=len(diff)
        if len(diff):print(name,'differences',len(diff),'first',diff[:4].tolist())
    assert np.all(got['counts'].to_numpy()==nl)
    print('authentic-table wrapper words',count,'differences',bad)
    if bad:raise SystemExit(1)
    for name, source in [('uflx','totuflux'),('dflx','totdflux'),('uflxc','totuclfl'),('dflxc','totdclfl')]:
        ref=get(name);got_host=result[source][0];idx=np.flatnonzero(ref.view(np.uint32)!=got_host.view(np.uint32));print(name,'host-vs-Fortran',len(idx),[(int(i),float(got_host[i]),float(ref[i])) for i in idx[:5]],flush=True)
