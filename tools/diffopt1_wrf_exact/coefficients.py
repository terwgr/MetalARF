"""Compare strict coordinate mixing coefficients with the selected CUDA body."""
import hashlib
import json
import re
import subprocess
from types import SimpleNamespace

import numpy as np
from tools.mp28_metal.parse import closure, split_top
from tools.smag_wrf_exact.compare import compare_words
from tools.thompson_real_column_parity import host_backend

host_backend.install()
from woof.core import kernels
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.metal.diff_opt1 import coefficients

source=kernels.module_source('diff_opt1')
expanded=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],
                        input=source,text=True,capture_output=True,check=True).stdout
items,_=closure(split_top(expanded),('wrf_diff_opt1_km4',))
selected='\n'.join(items)
host_source=re.sub(r'\b(bx|by)\b',lambda m:'boundary_'+m.group(),selected)
kernel=host_backend.HostModule('strict_coordinate_coefficients',host_source).get_function('wrf_diff_opt1_km4')
shape=(5,7,8); dims=[np.int32(n) for n in shape]; report={}
with MetalContext() as context:
    xp=MetalArrayModule(context)
    for name,dx,cs,scale in (('normal',1000.,.25,.01),('subnormal_minimum',1e-13,1e-6,1.)):
        d11=np.full(shape,scale,np.float32);d22=np.zeros(shape,np.float32)
        d12=np.zeros(shape,np.float32);maps=np.ones(shape[1:],np.float32)
        km=np.zeros(shape,np.float32);kh=np.zeros(shape,np.float32)
        pr=np.float32(1./3.)
        kernel(((shape[2]+127)//128,shape[1],shape[0]),(128,1,1),
               (d11,d22,d12,maps,np.float32(dx),np.float32(dx),np.float32(cs),pr,
                km,kh,*dims,np.int32(0),np.int32(0)))
        dev=lambda a:xp.asarray(a)
        state=SimpleNamespace(msft=dev(maps));cfg=SimpleNamespace(dx=dx,dy=dx,c_s=cs,
            open_x=False,open_y=False,specified=False,nested=False)
        dkm=xp.zeros(shape,np.float32);dkh=xp.zeros(shape,np.float32)
        coefficients(state,cfg,dev(d11),dev(d22),dev(d12),dkm,dkh,pr)
        report[name]={'km':compare_words(dkm.to_numpy(),km),'kh':compare_words(dkh.to_numpy(),kh)}
    assert report['subnormal_minimum']['km']['subnormal_words'] > 0
    assert all(not row['different_words'] for case in report.values() for row in case.values())
print(json.dumps({'source_sha256':hashlib.sha256(source.encode()).hexdigest(),
                  'host_flags':list(host_backend.CXX_FLAGS),'cases':report},indent=2))
