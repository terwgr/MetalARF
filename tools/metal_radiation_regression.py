"""Serial synchronous Metal radiation regression against the NumPy authority."""
import json, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from woof.metal.runtime import MetalContext
original = MetalContext._submit_kernel_batch
def synchronous(self, *args, **kwargs):
    result = original(self, *args, **kwargs)
    if result is not None: result.wait()
    return result
MetalContext._submit_kernel_batch = synchronous
from woof.metal.array import MetalArrayModule
from woof.core import rrtmgp as r
from tests.physics_reference.npref import np_rrtmgp_lw_rte
from woof.metal.suites import configuration
from woof.core.grid import make_vertical_coord, make_base_state
from woof.core.state import init_at_rest
from woof.core.physics import initialize_physics
from types import SimpleNamespace
module = MetalArrayModule()
try:
    tables = r.load_gas_tables('lw')
    bands = tables.gpoint_bands
    shp = (2, 3, tables.ngpt)
    gas = np.full(shp, .05, np.float32)
    lay = np.full(shp, 1.5, np.float32)
    lev = np.full((2,4,tables.ngpt), 1.5, np.float32)
    sfc = np.full((2,tables.ngpt), 1.8, np.float32)
    emis = np.ones_like(sfc)
    # Cloud optical depth only outside band zero: the broken map loses it.
    cloud_tau = np.zeros((2,3,tables.nband), np.float32)
    cloud_tau[:,0,1:] = np.array([.5,2.],np.float32)[:,None]
    u = module.asarray
    cld = r.CloudOpticsResult(u(cloud_tau),u(np.zeros_like(cloud_tau)),u(np.zeros_like(cloud_tau)))
    fused = r.FusedCloudOptics(tables, cld, None)
    tau = u(gas)
    args = r._fused_cloud_args(fused,tau,*shp)
    np.testing.assert_array_equal(args[3].to_numpy().view(np.int32), bands)
    # Seed the other consumer's float cache and confirm it is undisturbed.
    float_map = u(bands.astype(np.float32))
    tables._metal_band_maps = {id(module.context):float_map}
    assert r._fused_cloud_args(fused,tau,*shp)[3] is args[3]
    np.testing.assert_array_equal(float_map.to_numpy(),bands.astype(np.float32))
    results = {}
    for label, optics in [('clear',None),('cloudy',fused)]:
        out = r._lw_rte(tau,u(lay),u(lev),u(sfc),u(emis),top_at_1=False,
                        out=None,incident_out=None,fused_cloud=optics)
        expected_tau = gas if optics is None else gas+cloud_tau[:,:,bands]
        ref = np_rrtmgp_lw_rte(expected_tau,lay,lev,sfc,emis,top_at_1=False)
        up,down = out.flux_up.to_numpy(),out.flux_dn.to_numpy()
        explicit = r._lw_rte(u(expected_tau),u(lay),u(lev),u(sfc),u(emis),
            top_at_1=False,out=None,incident_out=None)
        np.testing.assert_array_equal(up.view(np.uint32),explicit.flux_up.to_numpy().view(np.uint32))
        np.testing.assert_array_equal(down.view(np.uint32),explicit.flux_dn.to_numpy().view(np.uint32))
        for got,want in [(up,ref.flux_up),(down,ref.flux_dn)]:
            # Established float32-vs-float64 RTE bound in test_rrtmgp.py.
            np.testing.assert_allclose(got,want,rtol=2e-5,atol=8e-5)
        results[label] = down[:,0].tolist()
    assert all(a>b for a,b in zip(results['cloudy'],results['clear']))
    cfg = configuration((0,0,0,0,1,'periodic'))
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord,lambda z:300.+.003*np.asarray(z),cfg.p_surf,cfg.ztop)
    state = init_at_rest(cfg,coord,base,array_module=module)
    from woof.core.dudhia import DudhiaShortwaveRadiation
    from datetime import datetime,timezone
    radiation = DudhiaShortwaveRadiation(datetime(2026,1,1,12,tzinfo=timezone.utc),
        np.full((cfg.ny,cfg.nx),35.,np.float32),np.full((cfg.ny,cfg.nx),-100.,np.float32))
    initialize_physics(state,cfg,glw=300.,radiation=radiation)
    np.testing.assert_array_equal(state.physics.fields['glw'].to_numpy(),300.)
    print(json.dumps(dict(results,initial_glw=300.,band_cache_separation=True,synchronous=True)))
finally: module.context.close()
