"""Generate native case-1 initialization and turbulence shaders."""
import pathlib,runpy,sys
root=pathlib.Path.cwd();out=pathlib.Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True);gen_dir=pathlib.Path(__file__).resolve().parent
shared=runpy.run_path(str(gen_dir/'generate_length1.py'))['length']
for kind in ('initialize','turbulence25'):
 s=(root/f'tools/mpasphys_v842_bl_mynn/generate_{kind}.py').read_text()
 s=s.replace('root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))','root=Path.cwd();sys.path.insert(0,str(root))')
 start=s.index("length=body('f_mym_length');")
 end=s.index("init=clean(body('f_mym_initialize'))" if kind=='initialize' else "turb=clean(body('f_mym_turbulence'))",start)
 s=s[:start]+"length=shared_length\n"+s[end:]
 old='hexv842_mpas_bl_mynn_initialize_f32' if kind=='initialize' else 'hexv842_mpas_bl_mynn_turbulence25_f32'
 new='hexv842_mpas_bl_mynn_initialize_length1_f32' if kind=='initialize' else 'hexv842_mpas_bl_mynn_turbulence26_length1_f32'
 oldtype='Hexv842MpasBlMynnInitializeF32' if kind=='initialize' else 'Hexv842MpasBlMynnTurbulence25F32'
 newtype='Hexv842MpasBlMynnInitializeLength1F32' if kind=='initialize' else 'Hexv842MpasBlMynnTurbulence26Length1F32'
 s=s.replace(old,new).replace(oldtype,newtype)
 s=s.replace("'vt','vq','qtke')", "'vt','vq','qtke','thetaw','elblmin','elblavg','dlu','dld')") if kind=='initialize' else s.replace("'gm','gh','qtke')", "'gm','gh','qtke','thetaw','elblmin','elblavg','dlu','dld')")
 s=s.replace("(root/f'woof/metal/shaders/{name}.metal').write_text(text)","(out/f'{name}.metal').write_text(text)")
 s=s.replace("(root/f'woof/metal/signature_fragments/{name}.json').write_text", "(out/f'{name}.json').write_text")
 exec(compile(s,str(out/f'generated_{kind}.py'),'exec'),{'shared_length':shared,'out':out,'__file__':str(out/f'generated_{kind}.py')})
 print('generated',new)
