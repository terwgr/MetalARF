"""Complete ordinary-FP32 Metal caller for P3 one-category MP50."""

from __future__ import annotations

from .p3_entry import submit_p3_entry_prepare
from .p3_final import submit_p3_final
from .p3_homofreeze import submit_p3_homofreeze
from .p3_kloop1 import submit_p3_kloop1
from .p3_kloopmain import submit_p3_kloopmain
from .p3_saveold_precip import submit_p3_saveold_precip
from .p3_sed_cloud import submit_p3_sed_cloud
from .p3_sed_ice import submit_p3_sed_ice
from .p3_sed_rain import submit_p3_sed_rain


def run_p3_metal(fields: dict, scratch: dict, diagnostics: dict,
                 surfaces: dict, tables, *, dt: float, it: int,
                 log_predict_nc: bool = False,
                 clbfact_dep: float = 1.0,
                 clbfact_sub: float = 1.0,
                 exponent: float = 287.15 / 1005.0) -> None:
    """Run the full P3 one-category timestep in source order on Metal.

    All prognostic, diagnostic, scratch, surface, and table arrays are owned
    by the caller and remain on one Metal context. ``fields`` uses the names
    in :mod:`woof.core.p3_device`; ``scratch`` includes its carrier and
    sedimentation workspace slots plus the WRF theta handoff ``h_diabatic``.
    """
    fields = dict(fields)
    if "theta" not in fields and "th" in fields:
        fields["theta"] = fields.pop("th")
    flags = scratch["flags"]
    fields["ssat"].fill(0.0)
    submit_p3_entry_prepare(
        theta=fields["theta"], pressure=fields["pressure"], qv=fields["qv"],
        h_diabatic=scratch["h_diabatic"],
        temperature=scratch["temperature"], tmparr1=scratch["tmparr1"],
        zdbz=diagnostics["zdbz"], effc=diagnostics["effc"],
        effi=diagnostics["effi"], vmi=diagnostics["vmi"],
        di=diagnostics["di"], rhopo=diagnostics["rhopo"],
        prt_liq=surfaces["prt_liq"], prt_sol=surfaces["prt_sol"],
        flags=flags, exponent=exponent)

    submit_p3_kloop1(fields=fields, scratch=scratch, flags=flags,
                     log_predict_nc=log_predict_nc)
    submit_p3_kloopmain(
        fields=fields, scratch=scratch, flags=flags,
        itab=tables.itab, icoll=tables.itabcoll,
        revap=tables.revap_table, dt=dt, it=it,
        log_predict_nc=log_predict_nc,
        clbfact_dep=clbfact_dep, clbfact_sub=clbfact_sub)
    submit_p3_sed_cloud(
        fields=fields, scratch=scratch, prt_liq=surfaces["prt_liq"],
        flags=flags, dt=dt, log_predict_nc=log_predict_nc)
    submit_p3_sed_rain(
        fields=fields, scratch=scratch, vm_table=tables.vm_table,
        vn_table=tables.vn_table, prt_liq=surfaces["prt_liq"],
        flags=flags, dt=dt)
    submit_p3_sed_ice(
        fields=fields, scratch=scratch, itab=tables.itab,
        prt_sol=surfaces["prt_sol"], flags=flags, dt=dt)
    submit_p3_homofreeze(fields=fields, scratch=scratch, flags=flags)
    submit_p3_final(fields=fields, scratch=scratch, diagnostics=diagnostics,
                    itab=tables.itab, flags=flags)
    submit_p3_saveold_precip(
        theta=fields["theta"], qv=fields["qv"],
        th_old=fields["th_old"], qv_old=fields["qv_old"],
        prt_liq=surfaces["prt_liq"], prt_sol=surfaces["prt_sol"],
        rainnc=surfaces["rainnc"], rainncv=surfaces["rainncv"],
        sr=surfaces["sr"], snownc=surfaces["snownc"],
        snowncv=surfaces["snowncv"], dt=dt)


__all__ = ["run_p3_metal"]
