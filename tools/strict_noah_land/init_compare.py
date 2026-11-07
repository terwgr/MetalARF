"""Verify the source-inactive Noah initialization branch and its refusal."""
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/thompson_real_column_parity'))
import host_backend
from woof.core import kernels
from woof.core.noah import load_tables,pack_params,initialize_noah_liquid_water
from woof.metal.array import MetalArrayModule
from woof.metal.runtime import MetalContext
from woof.wrf_exact import ENABLED

assert ENABLED
source='#define __launch_bounds__(...)\n'+kernels.module_source('noah_init')
k=host_backend.HostModule('strict_noah_init',source).get_function('noah_initialize_sh2o')
p=pack_params(load_tables())
soil=p.soil.copy(); soil[1,0]=0.0
p=replace(p,soil=soil)
shape=(4,2,2); categories=np.array([[1,1],[2,2]],np.int32)
temperature=np.full(shape,280.,np.float32);temperature[:,:,1]=272.;temperature[:,0,1]=273.149
moisture=np.linspace(.1,.4,16,dtype=np.float32).reshape(shape);moisture.flat[0]=np.array([1],np.uint32).view(np.float32)[0]
expected=np.full(shape,-99.,np.float32)
tables=np.asarray(p.soil,dtype=np.float32).reshape(p.slcats,10)
k((1,),(4,),[categories,moisture,temperature,expected,tables,np.int32(p.slcats),np.int32(4)])
with MetalContext() as context:
    xp=MetalArrayModule(context)
    dev={'isltyp':xp.asarray(categories),'smois':xp.asarray(moisture),'tslb':xp.asarray(temperature),'sh2o':xp.asarray(np.full(shape,-99.,np.float32))}
    initialize_noah_liquid_water(dev,p,(.1,.3,.6,1.))
    actual=dev['sh2o'].to_numpy()
    different=int(np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32)))
    assert different==0
    for mode in ('frozen','invalid'):
        cold=temperature.copy();cat=categories.copy()
        if mode=='frozen':cold[0,0,0]=272.
        else:cat[0,0]=np.iinfo(np.int32).min
        dev={'isltyp':xp.asarray(cat),'smois':xp.asarray(moisture),'tslb':xp.asarray(cold),'sh2o':xp.asarray(np.full(shape,-99.,np.float32))}
        try:initialize_noah_liquid_water(dev,p,(.1,.3,.6,1.))
        except (NotImplementedError if mode=='frozen' else ValueError):pass
        else:raise AssertionError(mode+' branch accepted')
        assert np.all(dev['sh2o'].to_numpy()==-99.)
print(f'words={expected.size} different_words={different}; frozen and invalid categories refused before SH2O mutation')
