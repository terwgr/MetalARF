#!/usr/bin/env python3
"""Exercise bundle refusal boundaries without loading corrupted native code."""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess, sys, tempfile, types
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from woof.metal.bundle_integrity import (BUNDLE_SCHEMA,BUNDLE_FILENAMES,
    metal_source_provenance,verify_metal_bundle)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--baseline-ref');a=p.parse_args()
 verify=verify_metal_bundle;source=metal_source_provenance;schema=BUNDLE_SCHEMA
 if a.baseline_ref:
  text=subprocess.check_output(['git','show',a.baseline_ref+':woof/metal/bundle_integrity.py'],cwd=ROOT,text=True)
  module=types.ModuleType('baseline');exec(compile(text,'<baseline-integrity>','exec'),module.__dict__)
  verify=module.verify_metal_bundle;source=lambda r:{'sha256':module.metal_source_sha256(r)};schema='woof-metal-bundle-v1'
 observed={}
 with tempfile.TemporaryDirectory(prefix='metal-bundle-gate-') as temp:
  root=pathlib.Path(temp)
  for name in ['woof/metal/kernel_signatures.json','woof/metal/shaders/sample.metal','woof/metal/signature_fragments/sample.json','woof/metal/native/woof_metal.mm','woof/metal/native/woof_metal.h','tools/build_metal_backend.py','tools/generate_metal_kernel_signatures.py','tools/metal_abi.py']:
   f=root/name;f.parent.mkdir(parents=True,exist_ok=True);f.write_text('fixture build input\n')
  bundle=root/'woof/libexec/metal';bundle.mkdir(parents=True)
  for name in BUNDLE_FILENAMES:(bundle/name).write_bytes(b'fixture artifact '+name.encode())
  manifest={'schema':schema,'metal_inputs_sha256':source(root)['sha256'],'source_provenance':source(root),'files':{name:{'size':(bundle/name).stat().st_size,'sha256':hashlib.sha256((bundle/name).read_bytes()).hexdigest()} for name in BUNDLE_FILENAMES}}
  mf=bundle/'METAL_BUNDLE.json';mf.write_text(json.dumps(manifest));verify(root)
  def check(label,mutation,restore):
   mutation()
   try:verify(root)
   except RuntimeError:observed[label]='refused'
   else:observed[label]='accepted'
   finally:restore()
  for name in BUNDLE_FILENAMES:
   f=bundle/name;original=f.read_bytes()
   check('changed-'+name,lambda f=f,b=original:f.write_bytes(bytes([b[0]^1])+b[1:]),lambda f=f,b=original:f.write_bytes(b))
   check('truncated-'+name,lambda f=f,b=original:f.write_bytes(b[:-1]),lambda f=f,b=original:f.write_bytes(b))
  for name in ['woof/metal/native/woof_metal.mm','woof/metal/native/woof_metal.h','tools/build_metal_backend.py','woof/metal/kernel_signatures.json']:
   f=root/name;original=f.read_bytes();check('changed-'+name,lambda f=f:f.write_bytes(b'changed input\n'),lambda f=f,b=original:f.write_bytes(b))
  f=root/'woof/metal/native/extra.h';check('added-native-input',lambda:f.write_text('extra'),lambda:f.unlink())
  f=bundle/BUNDLE_FILENAMES[0];b=f.read_bytes();check('missing-runtime',lambda:f.unlink(),lambda:f.write_bytes(b))
  if not a.baseline_ref:
   old=mf.read_bytes();check('legacy-manifest',lambda:mf.write_text(json.dumps(dict(manifest,schema='woof-metal-bundle-v1'))),lambda:mf.write_bytes(old))
   check('missing-source-binding',lambda:mf.write_text(json.dumps({k:v for k,v in manifest.items() if k!='source_provenance'})),lambda:mf.write_bytes(old))
   check('extra-asset-path',lambda:mf.write_text(json.dumps(dict(manifest,files=dict(manifest['files'],**{'../extra':{}})))),lambda:mf.write_bytes(old))
   verify(root);assert all(v=='refused' for v in observed.values()),observed
 print(json.dumps(observed,indent=2))
 if not a.baseline_ref:
  # Prove the real constructor rejects an overridden artifact before CDLL.
  import os
  from woof.metal import runtime
  original_loader=runtime.ctypes.CDLL;calls=[]
  def forbidden(path):calls.append(path);raise AssertionError('corrupted native loader reached')
  with tempfile.TemporaryDirectory(prefix='metal-override-') as temp:
   bad=pathlib.Path(temp)/'bad';bad.write_bytes(b'changed external artifact')
   try:
    runtime.ctypes.CDLL=forbidden
    for variable in ['WOOF_METAL_RUNTIME_LIB','WOOF_METAL_SHADER_LIBRARY','WOOF_METAL_EXACT_SHADER_LIBRARY']:
     previous=os.environ.get(variable);os.environ[variable]=str(bad)
     try:
      try:runtime.MetalContext()
      except RuntimeError as error:assert 'size or digest differs' in str(error),error
      else:raise AssertionError('corrupted override accepted')
     finally:
      if previous is None:os.environ.pop(variable,None)
      else:os.environ[variable]=previous
   finally:runtime.ctypes.CDLL=original_loader
  assert not calls
  print('All three overridden artifacts refused before native loading.')
  from tools.stage_wheel_metal import stage
  from tools.build_bridge_bundle import _metal_pin_record
  from woof import bridge_assets
  bundle=ROOT/'woof/libexec/metal'
  with tempfile.TemporaryDirectory(prefix='metal-staging-') as temp:
   staged=stage(bundle,pathlib.Path(temp))
   assert staged['schema']==BUNDLE_SCHEMA
  revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
  pin=_metal_pin_record(bundle/'METAL_BUNDLE.json',revision)
  payload={'schema':bridge_assets.PINS_SCHEMA,'release':'probe','platforms':{'darwin-arm64':{
   'bundle':{'filename':'probe.zip','bytes':1,'sha256':'0'*64},
   'binaries':[{'artifact':v.name,'filename':bridge_assets.artifact_filename(v,'darwin-arm64'),'bytes':1,'sha256':'0'*64} for v in bridge_assets.BUNDLED_ARTIFACTS],
   'assets':[],'metal':pin}}}
  decoded=bridge_assets.parse_pins(payload)
  assert decoded.platforms['darwin-arm64'].metal['source_inputs_sha256']==staged['metal_inputs_sha256']
  print('V2 Metal staging, release pinning and pin decoding passed.')
 return 0
if __name__=='__main__':raise SystemExit(main())
