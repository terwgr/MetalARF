"""Run the upper-band source comparison through the combined LW adapter."""
from pathlib import Path
path=Path(__file__).with_name('compare_upper.py')
source=path.read_text().replace(
    'from woof.hex.mpasphys.metal_rrtmg_lw_upper_v842 import band, BAND_TABLE_SIZES',
    'from woof.hex.mpasphys.metal_rrtmg_lw_v842 import band\n'
    'from woof.hex.mpasphys.metal_rrtmg_lw_upper_v842 import BAND_TABLE_SIZES')
exec(compile(source,str(path),'exec'),{'__file__':str(path),'__name__':'__main__'})
