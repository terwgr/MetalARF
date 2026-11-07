"""Compare ordered longwave tiers with the host-compiled source stages."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.goddard_metal import column_probe as probe
original_submit=probe._submit
original_words=probe.workspace_words
probe.workspace_words=lambda levels:original_words(levels)+1
def submit(name,resources,parameters,count):
    if name!='goddard_bands_columns':
        return original_submit(name,resources,parameters,count)
    for mode,band in [(0,0),*((1,i) for i in range(1,11)),(2,0)]:
        original_submit('goddard_tiers_columns',resources,
                        dict(parameters,tier_mode=mode,tier_band=band),count)
probe._submit=submit
raise SystemExit(probe.main())
