"""Find the netCDF library supplied by the installed Python wheel for child tools."""
import os
from pathlib import Path
import sys


def child_environment():
    env = os.environ.copy()
    if sys.platform in ("darwin", "linux") and not env.get("RW_MPAS_NETCDF_LIBRARY"):
        import netCDF4
        package = Path(netCDF4.__file__).parent
        if sys.platform == "darwin":
            libraries = sorted((package / ".dylibs").glob("libnetcdf*.dylib"))
        else:
            libraries = sorted((package.parent / "netCDF4.libs").glob("libnetcdf*.so*"))
        if libraries:
            env["RW_MPAS_NETCDF_LIBRARY"] = str(libraries[0])
    return env
