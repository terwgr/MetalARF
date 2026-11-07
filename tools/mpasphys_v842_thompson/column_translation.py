"""Translate the native Thompson step struct and audit its level workspace."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import re
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'tools/mp28_metal')]
from parse import split_top,def_name
from woof.hex.mpasphys import thompson_v842 as native


def translated_struct(specialization=native):
    source=specialization.cuda_source(device_math_source='// supplied by Metal support',max_levels=256,numerics='strict')
    source=source.replace('#define THOMPSON_COLUMN_KMAX 256','#define THOMPSON_COLUMN_KMAX TCKMAX',1)
    result=subprocess.run(['clang++','-E','-P','-std=c++17','-x','c++','-'],input=source.encode(),capture_output=True,check=True)
    items=split_top(result.stdout.decode())
    struct=next(item for item in items if def_name(item)=='ThompsonColumn')
    start=struct.index('__device__ __forceinline__ void calc_refl10cm(')
    end=native._body_end(struct,struct.index('{',start));struct=struct[:start]+struct[end:]
    start=struct.index('__device__ void rayleigh_soak_wetgraupel(')
    end=native._body_end(struct,struct.index('{',start));struct=struct[:start]+struct[end:]
    struct,count=re.subn(r'cd m_[iw]_0\s*=\s*\{[^}]*\};','',struct)
    if count!=2:raise ValueError('native radar-only complex fields changed')
    spec=spec_from_file_location('native_column_rewriter',ROOT/'tools/thompson_column_metal/generate.py')
    rewriter=module_from_spec(spec);spec.loader.exec_module(rewriter)
    struct=rewriter.eliminate_gotos(struct)
    constants,body=rewriter.convert_column_struct(struct)
    body,n=re.subn(r'^bool (\w+) \[ TCKMAX \] ;$',r'TCB ( \1 , TCKMAX ) ;',body,flags=re.M)
    counts=(body.count('TCF ('),body.count('TCD ('),n)
    expected=specialization.workspace_layout()['mpas_thompson_step']
    if counts!=(expected['float']-20,expected['double'],expected['bool']):
        raise ValueError(f'native Thompson step workspace changed: {counts}')
    if '[ TCKMAX' in body:raise ValueError('a level array escaped the native workspace')
    constants=re.sub(r"\bfloat\b","F",constants).replace("as_type<F>","as_type<float>")
    body=re.sub(r"\bfloat\b","F",body).replace("as_type<F>","as_type<float>")
    return constants,body,counts

if __name__=='__main__':
    constants,body,counts=translated_struct()
    print('native step method workspace float,double,bool:',counts)
    if len(sys.argv)>1:Path(sys.argv[1]).write_text(constants+'\n'+body)
