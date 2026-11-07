"""Transport selected native initialization columns into the Fortran oracle."""
import argparse
import struct
import netCDF4
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('input');p.add_argument('output');p.add_argument('--cells',type=int,nargs='+',required=True);p.add_argument('--julday',type=int,required=True)
a=p.parse_args()
with netCDF4.Dataset(a.input) as f, open(a.output,'wb') as out:
    n=f['rho'].shape[-1]
    if not 4<=n<=120:raise ValueError('native oracle needs 4 to 120 model levels')
    out.write(struct.pack('<4i',0x4c574331,n,len(a.cells),a.julday))
    for c in a.cells:
        out.write(struct.pack('<i',c))
        for name in ('rho','theta','qv','qc'):
            out.write(np.asarray(f[name][0,c,:],dtype='<f4').tobytes())
        out.write(np.asarray(f['zgrid'][c,:],dtype='<f4').tobytes())
        for name in ('skintemp','xland','xice','snow'):
            out.write(struct.pack('<f',float(f[name][0,c])))
        out.write(struct.pack('<f',float(f['latCell'][c])))
