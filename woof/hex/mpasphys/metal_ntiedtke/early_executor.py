"""Enqueue resident MPAS preparation through cloud-depth classification."""
import numpy as np
from .downstream_layouts import STAGES

SURFACE = set('ldcum ktype kcbot kctop kdpl lddraf idtop lndj pmfub wup zdhpbl scale_fac scale_fac2 zheat zcape zcape1 zcape2 ztauc ztaubl ztau_o upbl zmfub1 prfl prain prsfc pssfc zprecc raincv pratec'.split())
INTS = set('ldcum ktype kcbot kctop kdpl lddraf idtop lndj'.split())


def prepare(module, binding, constants, *, itimestep):
    fields=binding.fields
    shape=fields['t3d'].shape
    nz,ncol=shape; n=ncol; s=n*nz; si=n*(nz+1); st=n*(nz+2)
    ctx=module.context
    def new(words,dtype=np.float32): return module.zeros((words,),dtype=dtype)
    def flat(a): return a.view((a.size,))
    def view(a,start,length,dtype=None):
        return a.view((length,),dtype=dtype,byte_offset=a.byte_offset+4*start)
    def pack(destination, arrays):
        offset=0; pairs=[]
        for a in arrays:
            a=flat(a); pairs.append((a,view(destination,offset,a.size,a.dtype)));offset+=a.size
        ctx.submit_bookkeeping(pairs)
    def run(name,rs,ps,grid=n):
        ctx.submit_kernel('hexv842_nt_'+name,rs,ps,exact_policy=False,
                          grid_size=grid,threads_per_group=32)
    p=dict(ncol=n,nz=nz,**{x:np.float32(getattr(constants,x))
                           for x in ('cp','rd','rv','xlv','xls','xlf','grav')})
    delt=np.float32(np.float32(binding.dt_dyn)*np.float32(binding.stepcu))
    asc=dict(ncol=n,klev=nz,ztmst=delt,**{x:p[x] for x in ('cp','rd','rv','xlv','xls','xlf','grav')})
    flux=dict(p,tiedtke_closure=0,ztmst=delt)
    levels=new(12*s);interfaces=new(2*si);extra=new(4*s);surface=new(3*n)
    pack(levels,[fields[x] for x in ('t3d','pcps','dz8w','rho3d','qv3d','exner','rthdynten','rthblten','rthratenlw','rthratensw','rqvdynten','rqvblten')])
    pack(interfaces,[fields[x] for x in ('p8w','w')])
    pack(extra,[fields[x] for x in ('u3d','v3d','qc3d','qi3d')])
    pack(surface,[fields[x] for x in ('hfx','qfx','xland')])
    po=new(10*s);pio=new(3*si);math=new(3*n+s);env=new(12*st+3*n,np.uint32);saved=new(4*s)
    initflag=int(not binding.restart and itimestep==1)
    run('prefix',dict(levels=levels,interfaces=interfaces,out=po,outif=pio),dict(ncol=n,nz=nz,initflag=initflag,grav=p['grav']))
    run('prefix_math',dict(density=flat(fields['meshDensity']),level=levels,out=math),dict(ncol=n,nz=nz,len_disp=np.float32(binding.len_disp),rd=p['rd'],rv=p['rv']))
    run('provider_pack',dict(prefix=po,interfaces=pio,math=math,extra=extra,surface=surface,env=env,saved=saved),dict(ncol=n,nz=nz))
    ef=env.view((env.size,),dtype=np.float32); ei=env.view((env.size,),dtype=np.int32)
    ci=new(9*s);co=new(12*s);cii=new(s+n,np.int32)
    pack(ci,[view(ef,j*st+n,s) for j in range(9)])
    run('cuinin',dict(**{'in':ci},out=co,ints=cii),p)
    ti=new(9*st);to=new(3*st+n);toi=new(st+5*n,np.int32);ts=new(10*st);tsi=new(st,np.int32)
    def copy(pairs): ctx.submit_bookkeeping(pairs)
    pairs=[]
    tm=(1,-1,-2,8,7,6,2,9,0)
    for j,x in enumerate(tm):
        pairs.append((view(ef,x*st,st),view(ti,j*st,st)) if x>=0 else (view(co,(-x-1)*s,s),view(ti,j*st+n,s)))
    for j,x in enumerate((3,4,11)):pairs.append((view(co,x*s,s),view(to,j*st+n,s)))
    pairs.append((view(cii,0,s),view(toi,n,s)));copy(pairs)
    run('cutypen',dict(inputs=ti,surface=view(surface,0,2*n),outs=to,iouts=toi,scr=ts,scr_i=tsi),p)
    mi=new(10*st);mii=new(3*n,np.int32);mo=new(3*n);mld=new(n,np.int32)
    pairs=[(view(ef,x*st,st),view(mi,j*st,st)) for j,x in enumerate((10,11,7,3,4))]
    pairs += [(view(to,j*st,st),view(mi,(5+j)*st,st)) for j in range(3)]
    pairs += [(view(co,j*s,s),view(mi,(8+j)*st+n,s)) for j in range(2)]
    pairs += [(view(toi,st+n,2*n),view(mii,0,2*n)),(view(ei,12*st+2*n,n),view(mii,2*n,n)),(view(toi,st,n),mld)]
    copy(pairs);run('mfub',dict(inputs=mi,surf_i=mii,out_i=mld,out=mo),flux)
    ai=new(8*st);aa=new(2*n,np.int32);ao=new(13*st);af=new(st+5*n,np.int32);aout=new(2*n);start=new(nz+3,np.int32)
    pairs=[(view(ef,x*st,st),view(ai,j*st,st)) for j,x in enumerate((0,1,2,6,8,9,7,5))]
    pairs += [(view(ei,12*st+2*n,n),view(aa,0,n)),(view(toi,st+4*n,n),view(aa,n,n))]
    pairs += [(view(co,j*s,s),view(ao,j*st+n,s)) for j in range(2)]
    pairs += [(view(to,j*st,st),view(ao,(j+2)*st,st)) for j in range(3)]
    pairs += [(view(toi,0,st),view(af,0,st)),(mld,view(af,st,n)),(view(toi,st+n,2*n),view(af,st+n,2*n)),(view(toi,st+3*n,n),view(af,st+4*n,n)),(view(mo,2*n,n),view(aout,0,n))]
    copy(pairs)
    run('ascent',dict(ldcum=view(af,st,n),ktype=view(af,st+n,n),klab=view(af,0,st),pqen=view(ef,st,st),pqsen=view(ef,2*st,st),pgeo=view(ef,6*st,st),result=start),dict(ncol=n,klev=nz,grav=p['grav']),grid=1)
    run('cuascn',dict(inputs=ai,surf_in_i=aa,surf_in_f=view(to,3*st,n),levels=ao,flags=af,surf_out=aout,mpas_llo3_start=view(start,0,1)),asc)
    di=new(2*st);ds=new(3*n,np.int32);dout=new(5*st+n);dsi=new(2*n,np.int32)
    copy(((view(ef,7*st,st),view(di,0,st)),(view(ao,10*st,st),view(di,st,st)),(view(af,st,n),view(ds,0,n)),(view(af,st+2*n,2*n),view(ds,n,2*n)),(view(af,st+n,n),view(dsi,0,n)),(view(af,st+4*n,n),view(dsi,n,n))))
    run('cloud_depth',dict(**{'in':di},sf=ds,out=dout,sout=dsi),dict(ncol=n,klev=nz))
    names={name for stage in STAGES for group in stage['groups'] for name in group}|{'ztt','zqq','ztenu','ztenv'}
    state={x:new(n if x in SURFACE else st,np.int32 if x in INTS else np.float32) for x in names}
    pairs=[]
    def into(src,off,name,count,dest=0):pairs.append((view(src,off,count),view(state[name],dest,count)))
    for j,x in enumerate('pten pqen pqsen puen pven pverv pgeo paph pgeoh pap ptte pqte'.split()):
        if x in state:into(ef,j*st,x,st)
    for j,x in enumerate('ptenh pqenh ptu pqu plu pmfu pmfus pmfuq pmful plude pdmfup plglac pmfude_rate'.split()):into(ao,j*st,x,st)
    for j,x in enumerate(('ldcum','ktype','kcbot','kctop')):into(af,st+j*n,x,n)
    into(dsi,0,'ktype',n);into(toi,st+4*n,'kdpl',n);into(ei,12*st+2*n,'lndj',n)
    for src,off,x in ((aout,0,'pmfub'),(aout,n,'wup'),(mo,0,'zdhpbl'),(mo,n,'upbl'),(math,n,'scale_fac'),(math,2*n,'scale_fac2')):into(src,off,x,n)
    for j,x in enumerate(('pmfd','pmfds','pmfdq','pdmfdp','pdpmel')):into(dout,j*st,x,st)
    into(dout,5*st,'prfl',n)
    for j,x in ((5,'ptd'),(6,'pqd'),(7,'puu'),(8,'pvu'),(9,'pud'),(10,'pvd')):into(co,j*s,x,s,n)
    for j,x in enumerate(('qcf','qif','uf','vf')):into(saved,j*s,x,s,n)
    into(ef,0,'tf',st)
    for source,x in (('t3d','t'),('qv3d','qv'),('qc3d','qc'),('qi3d','qi'),('u3d','u'),('v3d','v'),('exner','exner')):pairs.append((flat(fields[source]),view(state[x],n,s)))
    into(ef,10*st,'ztt',st);into(ef,11*st,'zqq',st)
    # submit_bookkeeping accepts at most 64 pairs.
    for start_pair in range(0,len(pairs),64):copy(pairs[start_pair:start_pair+64])
    return state
