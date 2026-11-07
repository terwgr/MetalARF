"""Publish the pinned RRFS RUC startup state to native Metal pools."""

import numpy as np

from ..resources_build import load_packet
from .framework_init_rrfs import FrameworkInitializationRRFS
from .metal_ruc_boundaries import initialize_columns
from .metal_ruc_tables import decode_pinned_ruc_tables
from .ruc_lsm_8c9f1a8 import TABLE_LAYOUT


_INPUTS = {name: "sfc_input" for name in ("ivgtyp", "isltyp", "xice", "tslb", "smois")}
_OUTPUTS = {"sh2o": "sfc_input", "smfr3d": "diag_physics", "mavail": "diag_physics", "znt": "diag_physics"}


def initialize_ruc(session):
    """Initialize full local columns, then publish results and native table words.

    The solve-only pool views keep their existing layout. On restart or DA
    cycling, only the source's table initialization runs.
    """
    native, pools, layout = session.native, session.pools, session.layout
    classification = native.get_text_field("sfc_input", "mminlu")
    landuse = FrameworkInitializationRRFS.RUC_LANDUSE.get(classification)
    if landuse is None:
        raise ValueError(f"RRFS RUC has no table for land-use classification {classification!r}")
    if native.resources is None:
        raise RuntimeError("RRFS Metal RUC initialization requires the route resource packet")
    table_directory = load_packet(native.resources)["sf_noah"]["table_directory"]
    words = decode_pinned_ruc_tables(table_directory, landuse)
    tables = pools.arrays.asarray(words.view(np.float32))

    configs = layout.configs
    if not (configs["config_do_restart"] or configs["config_do_DAcycling"]):
        cells = layout.dimensions["nCells"]
        fields = {name: native.field_storage(pool, name)
                  for name, pool in (_INPUTS | _OUTPUTS).items()}
        if fields["tslb"].shape[0] != layout.dimensions["nSoilLevels"]:
            raise ValueError("RRFS RUC startup soil storage differs from the native Registry")
        if any(value.shape[-1] < cells for value in fields.values()):
            raise ValueError("RRFS RUC startup requires all local columns")
        context = pools.arrays.context
        size = sum(value[..., :cells].nbytes for value in fields.values())
        if size > context.recommended_working_set_size:
            raise MemoryError("RRFS RUC startup workspace exceeds the Metal device budget")
        work = {name: pools.arrays.asarray(np.ascontiguousarray(value[..., :cells]))
                for name, value in fields.items()}
        initialize_columns(work, tables, solve_cells=cells, restart=False, cycling=False)
        results = {name: work[name].to_numpy() for name in _OUTPUTS}
        for name, pool in _OUTPUTS.items():
            fields[name][..., :cells] = results[name]
            cached = pools.buffers.get((pool, name, 1))
            if cached is not None:
                _, device, selection = cached
                device[...] = np.ascontiguousarray(fields[name][selection])

    for name, spec in TABLE_LAYOUT.items():
        if spec["character"]:
            continue
        start = spec["offset"]
        value = words[start:start + spec["words"]].view(np.int32 if spec["type"] == "int" else np.float32)
        native.put_table("module_ruc_land." + name, value if spec["dim"] else value[0],
                         lower=(1,) * len(spec["dim"]))
    native.scheme_state["ruc_table_words"] = tables
