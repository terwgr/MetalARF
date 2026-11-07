"""A preparation never imports netCDF4 (D-10, 2.8.8 acceptance).

THE BREAKAGE (c8f95278, every row the site's ``engine_prepare.py``
prepared): that preparer hosts the engine's preparation chains in its own
process, which the command line's ``PYTHON_GIL=0`` re-exec cannot reach,
and each ``prep.log`` read "The global interpreter lock (GIL) has been
enabled to load module 'netCDF4._netCDF4'".  No byte was read through
netCDF4: the parser (``woof.downscale`` via ``woof.offline_child``), the
domain-artifact reader (``woof.wrf_direct``) and the memory estimate
(``woof.core.rrtmgp``) imported it at module scope.

Every check runs in a fresh interpreter, because the test process itself
may have imported netCDF4 long before.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

#: What the in-process preparation reaches, traced on the acceptance rows
#: (gfs, hrrr, icon-global): the chain, the CLI parser its fetch builds,
#: the stage-reuse binding, the memory estimate, and every module that
#: imported netCDF4 at module scope before D-10.
_PREPARATION_PATH = '''
import json, sys
import woof.runplan, woof.regional_preparation, woof.experiment
import woof.go_cli, woof.stage_reuse, woof.core.preflight
import woof.wrf_direct, woof.downscale, woof.offline_child
import woof.offline_child_run, woof.io.wrfout, woof.core.rrtmgp
import woof.gfs_direct, woof.era5_direct, woof.mapped_direct
import woof.native_domain_artifacts, woof.native_hierarchy
from woof import cli
cli.build_parser()
woof.core.rrtmgp.rrtmgp_above_model_layer_counts(5000.0)
probe = getattr(sys, "_is_gil_enabled", None)
print(json.dumps({"netCDF4": "netCDF4" in sys.modules,
                  "gil": None if probe is None else bool(probe())}))
'''


def _fresh(code: str, *args: str) -> dict:
    env = {key: value for key, value in os.environ.items()
           if key not in ("PYTHON_GIL", "GPUWM_FREE_THREADING_REEXEC")}
    env["PYTHONPATH"] = str(REPO)
    completed = subprocess.run([sys.executable, "-c", code, *args],
                               capture_output=True, text=True, env=env,
                               cwd=REPO, timeout=600)
    assert completed.returncode == 0, completed.stderr[-4000:]
    assert "GIL) has been enabled" not in completed.stderr, completed.stderr[-4000:]
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_preparation_path_imports_no_netcdf4():
    found = _fresh(_PREPARATION_PATH)
    assert found["netCDF4"] is False
    # On the free-threaded build the lock stays off, which is the claim
    # the acceptance row checks.
    import sysconfig

    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        assert found["gil"] is False


def test_memory_estimate_reads_rrtmgp_tables_without_netcdf4():
    """The tables the memory estimate packs come through rw_netcdf.

    With the module-scope imports gone, this was the read that still
    switched the lock on in every site-prepared row: the estimate's
    ``k_distribution_bytes`` packs the gas and cloud tables.
    """

    from woof.netcdf_bridge import find_netcdf_bin

    if find_netcdf_bin() is None:
        pytest.skip("rw_netcdf is not built in this tree")
    try:
        from woof.core.rrtmgp import _table

        for name in ("rrtmgp-gas-lw-g256.nc", "rrtmgp-gas-sw-g224.nc",
                     "rrtmgp-clouds-lw-bnd.nc", "rrtmgp-clouds-sw-bnd.nc"):
            _table(name)
    except Exception as error:                          # noqa: BLE001
        pytest.skip(f"RRTMGP tables are not staged here: {error}")
    found = _fresh('''
import json, sys
from woof.core import rrtmgp
for kind in ("lw", "sw"):
    tables = rrtmgp.load_gas_tables(kind)
    clouds = rrtmgp.load_cloud_tables(kind)
    assert tables.kmajor.dtype.name == "float64" and tables.ngpt > 0
    assert clouds.extliq.dtype.name == "float64"
    assert "h2o" in tables.gas_names
probe = getattr(sys, "_is_gil_enabled", None)
print(json.dumps({"netCDF4": "netCDF4" in sys.modules,
                  "gil": None if probe is None else bool(probe())}))
''')
    assert found["netCDF4"] is False
    import sysconfig

    if sysconfig.get_config_var("Py_GIL_DISABLED"):
        assert found["gil"] is False


def test_deferred_module_imports_on_first_use():
    code = '''
import json, sys
from woof.io.netcdf_serialization import netCDF4
before = "netCDF4" in sys.modules
Dataset = netCDF4.Dataset
print(json.dumps({"before": before, "after": "netCDF4" in sys.modules,
                  "real": Dataset is sys.modules["netCDF4"].Dataset}))
'''
    env = {key: value for key, value in os.environ.items() if key != "PYTHON_GIL"}
    env["PYTHONPATH"] = str(REPO)
    completed = subprocess.run([sys.executable, "-W", "ignore", "-c", code],
                               capture_output=True, text=True, env=env,
                               cwd=REPO, timeout=600)
    assert completed.returncode == 0, completed.stderr[-4000:]
    found = json.loads(completed.stdout.strip().splitlines()[-1])
    assert found == {"before": False, "after": True, "real": True}


def test_run_stamp_reads_the_init_time_through_rust(tmp_path):
    """``wrfout_init`` reads SIMULATION_START_DATE on the Rust route."""

    netCDF4 = pytest.importorskip("netCDF4")
    from woof.netcdf_bridge import find_netcdf_bin

    if find_netcdf_bin() is None:
        pytest.skip("rw_netcdf is not built in this tree")
    path = tmp_path / "wrfout_d01_2026-10-08_12:00:00"
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as dataset:
        dataset.createDimension("Time", None)
        # One record, as every real history has; the decoder refuses an
        # unlimited dimension it cannot size.
        dataset.createVariable("XTIME", "f4", ("Time",))[0] = 60.0
        dataset.setncattr("SIMULATION_START_DATE", "2026-10-08_12:00:00")
    found = _fresh('''
import json, sys
from woof import run_stamp
value = run_stamp.wrfout_init(sys.argv[1])
print(json.dumps({"init": None if value is None else value.isoformat(),
                  "netCDF4": "netCDF4" in sys.modules}))
''', str(path))
    assert found["netCDF4"] is False
    assert found["init"] is not None and found["init"].startswith("2026-10-08T12:00:00")
