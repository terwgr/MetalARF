"""Pure, fail-closed options for the built-in Arwen CUDA column suite."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import math

from ..core.ysu_contract import YSU_FREE_ATMOSPHERE_MIXING_LENGTHS

from ..constants import NATIVE_PHYSICS_ACKNOWLEDGEMENT


# Option value -> the name the component order reports for the slot.
CUMULUS_COMPONENTS = {"gf": "gf", "own": "arwen-massflux-v1", "ntiedtke": "ntiedtke"}


@dataclass(frozen=True)
class NativePhysicsOptions:
    acknowledgement: str
    start_time_utc: str
    radiation: str = "rrtmgp"
    radiation_interval_s: float = 1800.0
    radiation_column_chunk: int = 12_500
    radiation_validation_mode: str = "fused"
    surface_layer: str = "sfclay"
    sfclay_option: int = 1
    # Land thermal-roughness closure of the surface layer (sfclay.cu
    # iz0tlnd): 0 the Carlson-Boland scalar roughness WRF runs by default
    # (z0h fixed near 4e-4 m under the viscous term, whatever the canopy),
    # 1 Chen-Zhang 2009, 2 Zilitinkevich with czil 0.1, 3 Zilitinkevich
    # with the bare-soil czil 0.8 weighted by the square of the bare
    # fraction (Zheng et al. 2012, the GFS form): a vegetated column
    # exchanges heat with z0h near z0m, bare soil with a large kB^-1.
    # Read under 3 from the surface state's vegetation fraction.  Outside
    # 0 the option joins the identity; at 0 no arithmetic changes and a
    # checkpoint keeps its hash.
    sfclay_iz0tlnd: int = 0
    land_surface: str = "noah"
    land_surface_interval_s: float = 60.0
    pbl: str = "ysu"
    ysu_topdown_pblmix: int = 1
    # The asymptotic mixing length of YSU's local diffusivity above the
    # boundary layer (woof.globe.core.ysu_contract).  WRF ties it to the layer
    # thickness (0.1 dz clipped to 30..300 m, bl_ysu.F90:1003), so on this
    # suite's 40-level stack, whose layers across the jet are 55 hPa (about
    # 1500 m) thick, the length reads 150 m and the diffusivity 25x what a
    # 300 m regional layer gets for the same shear and Richardson number;
    # measured on the 2026-09-01 00Z T255 control the physics drained the
    # 100 to 400 km kinetic energy at 237 hPa at 0.7 to 1.2 per day, 99.6
    # percent of it this operator (the energy ledger).
    # "wrf-layer" is WRF's rule, bit for bit (the regional model's only
    # setting) and the DEFAULT here.  "fixed" holds the length at WRF's own
    # rlam = 30 m at every spacing, the value WRF's rule gives every layer
    # thinner than 300 m; nothing inside the boundary layer changes.  It is
    # selectable, not default, and is reported as a workaround: its 24 h
    # T255 grade (2026-09-05) lifted the northern 250 km energy ratio at
    # 250 hPa from 0.67 to 0.85 and improved the 250 hPa speed bias, but
    # worsened the northern 250 hPa vector wind rmse against the GFS f024
    # by 0.16 m/s (4.41 to 4.56, the gap opening from hour 9) and lifted
    # the grid-limit ratio above the diffusion package's (0.053 against
    # 0.038); the rule that admits a default asked for both not to worsen.
    # Identity: the flag joins the config identity only as "fixed" (see
    # ``identity``), so every checkpoint written under WRF's rule keeps
    # its hash, bare or with "wrf-layer" spelled.
    ysu_free_atmosphere_mixing_length: str = "wrf-layer"
    # Cumulus parameterization, default ON (fixed means default).  The
    # suite runs at 25-52 km, where the resolved grid cannot carry deep
    # convection: without a scheme every convective column waits for
    # grid-scale saturation and rains through Morrison as a grid-point
    # storm (audit 2026-09-01, the sixth native component).  "gf" is the
    # scale-aware WRF-v4.6.1 Grell-Freitas kernel behind woof.globe.core.gf,
    # run between YSU and Morrison in WRF's order; "ntiedtke" is the
    # WRF-v4.6.1 New Tiedtke scheme behind woof.globe.core.ntiedtke (the
    # regional cu_physics = 16, bitwise against the frozen Fortran at all
    # 21 stages), in the same slot; "none" is the research arm without
    # either.  gf_ishallow follows the MPAS seam
    # (woof/core/mpas_column_batch.py: native MPAS hardwires ishallow=1),
    # read only where cumulus="gf"; the closure is the 16-member ensemble
    # (clos_choice=0), the only arm with oracle coverage, and is not an
    # option here.  "own" is Arwen Global's own closure, arwen-massflux-v1
    # (woof/globe/physics/arwen_massflux.py): a scale-aware
    # deep/shallow mass-flux scheme written for this Gaussian grid, this
    # 40-level split and this step, with convective momentum transport,
    # in the same slot.
    cumulus: str = "gf"
    gf_ishallow: int = 1
    # Read only where cumulus="gf".  WRF's GFDRV rejects a column whose
    # downdraft cannot form (not negatively buoyant, or nothing to
    # evaporate: every deep, saturated column) and hands it to the grid
    # scale; on this 50 km column the grid scale cannot carry deep
    # convection and the hand-off is the grid-point storm (the warm-pool
    # cell of the 2026-09-01 control: 1,200 J/kg of CAPE, seventeen
    # saturated levels, 84 mm/h of grid-scale rain beside 0.06 mm/h from
    # the scheme; the ierr 7 exits and cup_dd_moisture's ierr 51
    # zero-denominator exit, which the control's 311 mm central-Pacific
    # storm cell at 13.3N 211E carried on all 24 calls of its hour-12
    # trace).  True runs the deep arm updraft-only in
    # those columns (gf.cu GF_UPDRAFT_ONLY_WHEN_DOWNDRAFT_DRY); False is
    # WRF's kernel word for word and is the identity every Grell-Freitas
    # checkpoint before this option existed was pinned with, so only True
    # joins the hash.  Default False by the timing lane's grade of
    # 2026-09-05 on the 2026-09-01 00Z T255 case (both arms carrying the
    # advective forcing lanes): the switch halves the grid-scale cells
    # above 200 mm/day (19 to 12 on the export grid, 27 on the control)
    # and lowers the CONUS T2 and MSLP errors by 0.01 K and 0.004 hPa
    # beyond the lanes alone, but takes the northern 250 km energy ratio
    # from 0.806 to 0.791 and the grid-limit ratio from 0.561 to 0.545
    # (0.825 / 0.584 on the control: 0.034 / 0.039 below it, past the
    # 0.03 bar the grade set) and leaves the CONUS diurnal composite at
    # 12.93 LST against Stage-IV's 14.60, so it stays an opt-in until a
    # remedy meets the bar; the forcing lanes ship by themselves.
    gf_updraft_only_when_downdraft_dry: bool = False
    # Read only where cumulus="gf".  The coarse-column form of the deep
    # arm (gf.cu GF_RESOLVED_CONVERGENCE_CLOSURE), a declared divergence
    # from WRF v4.6.1 for a 52 km column that carries the whole of deep
    # convection as resolved forcing: (1) neg_check's 300.01 K/day
    # per-level heating cap, a bound tuned where the grid resolves part of
    # the convection, yields to the latent heat of the column's own
    # resolved moisture convergence (the cap becomes the larger of the
    # two); (2) the downdraft sweeps continue past levels the downdraft's
    # beta profile reaches with no mass, carrying the environment there,
    # where WRF divides by zero and exits with ierr 51 (the control's
    # central-Pacific storm at 13.3N 211E on all 24 calls of its hour-12
    # trace, 10,900 columns per call over the planet); (3) a downdraft that
    # cannot form (ierr 7) leaves the updraft running, the
    # gf_updraft_only_when_downdraft_dry behaviour; (4) the cloud-base mass
    # flux is floored at the ensemble's own Kuo moisture-convergence member
    # (a quarter of the sixteen-member mean's weight in WRF) times the
    # Kuo-Anthes share of the converged moisture that rains out (1 - b,
    # b = 10 (1 - RH) clipped to 0..1, RH the cloud layer's mean relative
    # humidity: nothing under 90 percent, all of it at saturation), so a
    # saturated convecting column precipitates the moisture the grid
    # converges into it instead of handing it to the microphysics as a
    # grid-point storm, and an unsaturated one moistens as WRF's members
    # already do.  True joins the hash
    # (a different trajectory); False is WRF's kernel word for word and
    # keeps the identity every earlier Grell-Freitas checkpoint carried.
    # Default False by the mass-flux lane's grade of 2026-09-05 on the
    # 2026-09-01 00Z T255 case (three arms, this form the last): it takes
    # the grid-scale cells above 200 mm/day from 27 to 0 and the heaviest
    # from 448 to 177 mm with every surface and upper-air score inside the
    # 0.03 bar, but takes the northern 250 km energy ratio from 0.825 to
    # 0.792 (0.034, past the 0.03 bar the grade set), the global grid-scale
    # rain from 1.500 to 1.187 mm/day (0.313, past the 0.3 bar) and the
    # CONUS diurnal composite only to 13.00 LST against Stage-IV's 14.60
    # (the bar was within 1 h), so it is an OPT-IN, a workaround for the
    # grid-point storms until a form meets the bar; the closure reading and
    # its census ship with either value.  The surface gains of that grade
    # (CONUS T2 rmse 2.548 to 2.481 K, MSLP rmse 1.676 to 1.575 hPa) are
    # the dynamics forcing lanes', not the arm's: with the arm off on the
    # same tree they read 2.477 K and 1.588 hPa, the northern 250 km ratio
    # 0.806 and the grid-scale rain 1.418 mm/day.  The floor is compared
    # with the mean after the diurnal-cycle term is taken off it, so a
    # column that term silenced convects on the floor alone (1,298 of
    # 15,357 deep-active columns at the arm's hour 12; the census names
    # them).
    gf_resolved_convergence_closure: bool = False
    # Read only where cumulus="ntiedtke": True runs the New Tiedtke kernels
    # with classic Tiedtke's deep closure (the fixed 2400 s adjustment time
    # and the moisture-convergence first guess in place of the
    # resolution-scaled time and the geometric fraction;
    # docs/cumulus-new-tiedtke.md).  False leaves every result bit-identical
    # to the port.  Outside cumulus="ntiedtke" the flag is not part of the
    # identity (see ``identity``), so a Grell-Freitas checkpoint keeps its
    # hash.
    ntiedtke_tiedtke_closure: bool = False
    # Columns the Grell-Freitas seam packs per pass (woof.globe.core.gf
    # GF_COLUMN_CHUNK).  MEASURED on the T533 40-level batch (1,283,202
    # columns, RTX 5090 32 GB): packed whole, the scheme's input and output
    # blocks were 3.07 + 3.28 GB and the run died at its first cumulus
    # call.  Column-independent, so the chunk moves memory and nothing else.
    cumulus_column_chunk: int = 131_072
    microphysics: str = "morrison"
    morr_rimed_ice: int = 1
    dx_m: float = 50_000.0
    # Measured and reported, never enforced: the suite writes
    # native_water_residual_exceeds_tolerance into its diagnostics and no gate
    # consumes it.  validate() therefore refuses a caller-chosen value rather
    # than accepting a threshold that does nothing.
    water_fix_tolerance_kg_m2: float = 1.0e-5
    energy_change_advisory_j_m2: float = 5.0e4
    initial_pbl_height_m: float = 800.0
    initial_soil_temperature_k: float = 285.0
    initial_soil_water_fraction: float = 0.30
    # The per-column vegetation and soil categories live in the surface
    # state (woof.globe.statics: real WPS_GEOG fields, or the
    # declared synthetic planet); the former vegetation_category /
    # soil_category constants here were that planet and are retired.
    land_ice_category: int = 15
    urban_category: int = 13
    use_monthly_albedo: bool = False
    read_lai_2d: bool = True
    noah_thermal_conductivity_option: int = 1
    wrf_rrtmg_compatibility: str = "none"
    trace_co2_ppm: float = 369.55
    # Stratospheric cold-top floor (default ON).  Defect: the 384 h native
    # baseline arm (T255, GDAS 2026-09-01 00Z) died at hour 83.4 --
    # "temperature outside research bounds: 139.995..298.852 K" -- via a slow
    # polar-night top sag (~5e-5 K/s, 144.2 -> 141.7 K across the final
    # diagnostics) with winds healthy.  RRTMGP carries ozone shortwave, but a
    # 20-level hydrostatic top cannot supply the residual-circulation warming
    # that holds the real polar-night stratopause near 200-230 K at 1-3 hPa,
    # so the top relaxes radiatively downward without bound.  Same
    # Held-Suarez-style scaffold the reference suite has carried since v5
    # (reference.py _stratospheric_floor_pull), same defaults; against the
    # measured sag rate the one-sided pull balances 0.09 K below the floor.
    # stratospheric_floor_k <= 0 disables (research arms measuring the
    # unfloored sag).
    stratospheric_floor_k: float = 195.0
    stratospheric_floor_pa: float = 5_000.0
    stratospheric_relaxation_time_s: float = 1_800.0

    @classmethod
    def from_mapping(cls, raw: dict[str, object]) -> "NativePhysicsOptions":
        if not isinstance(raw, dict):
            raise TypeError("native adapter options must be a mapping")
        unknown = sorted(set(raw) - set(cls.__dataclass_fields__))
        if unknown:
            raise ValueError(
                "unknown native adapter options: " + ", ".join(unknown)
            )
        try:
            value = cls(**raw)
        except TypeError as exc:
            raise ValueError(f"invalid native adapter options: {exc}") from exc
        value.validate()
        return value

    @property
    def normalized(self) -> dict[str, object]:
        """Every option, validated, with its default filled in: what the
        suite is built from.  ``from_mapping(normalized)`` rebuilds these
        options exactly; ``from_mapping(identity)`` does NOT (an option the
        identity drops comes back as its default), so nothing that runs a
        scheme may be built from the identity."""
        return asdict(self)

    @property
    def identity(self) -> dict[str, object]:
        payload = asdict(self)
        # The identity carries every normalised option and the config hash
        # (and so every checkpoint's restart admission) is built from it.
        # A scheme-scoped flag joins only under the scheme that reads it:
        # under cumulus="gf" or "none" the New Tiedtke closure flag changes
        # no arithmetic, so it must not move the hash of a Grell-Freitas
        # checkpoint (tests/test_arwen_global_vertical_surface_stretched.py
        # pins that hash), while under cumulus="ntiedtke" the two closures
        # are different trajectories and never share one.
        if self.cumulus_scheme != "ntiedtke":
            del payload["ntiedtke_tiedtke_closure"]
        # WRF's thickness-scaled length is the arithmetic every checkpoint
        # before 2026-09-05 was written under, when the option did not
        # exist: spelling it must keep those hashes.  "fixed" is a
        # different trajectory and joins.
        if self.ysu_free_atmosphere_mixing_length == "wrf-layer":
            del payload["ysu_free_atmosphere_mixing_length"]
        if self.cumulus_scheme != "gf" or not self.gf_updraft_only_when_downdraft_dry:
            # The WRF-faithful state keeps the identity Grell-Freitas
            # checkpoints carried before the option existed.
            del payload["gf_updraft_only_when_downdraft_dry"]
        if self.cumulus_scheme != "gf" or not self.gf_resolved_convergence_closure:
            del payload["gf_resolved_convergence_closure"]
        if self.sfclay_iz0tlnd == 0:
            del payload["sfclay_iz0tlnd"]
        return payload

    @property
    def cumulus_enabled(self) -> bool:
        return self.cumulus_scheme is not None

    @property
    def cumulus_scheme(self) -> str | None:
        """``"gf"``, ``"ntiedtke"``, ``"own"`` or None for cumulus="none"."""
        name = str(self.cumulus).lower()
        return None if name == "none" else name

    @property
    def uses_grell_freitas(self) -> bool:
        return self.cumulus_scheme == "gf"

    @property
    def cumulus_component(self) -> str | None:
        """The name the component order carries for the cumulus slot."""
        scheme = self.cumulus_scheme
        return None if scheme is None else CUMULUS_COMPONENTS[scheme]

    @property
    def start_time(self) -> datetime:
        text = self.start_time_utc
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        value = datetime.fromisoformat(text)
        if value.tzinfo is None:
            raise ValueError("start_time_utc must carry an explicit UTC offset")
        return value.astimezone(timezone.utc)

    def validate(self) -> None:
        if self.acknowledgement != NATIVE_PHYSICS_ACKNOWLEDGEMENT:
            raise ValueError(
                "native adapter acknowledgement must be exactly "
                f"{NATIVE_PHYSICS_ACKNOWLEDGEMENT!r}"
            )
        _ = self.start_time
        required = {
            "radiation": (self.radiation, "rrtmgp"),
            "surface_layer": (self.surface_layer, "sfclay"),
            "land_surface": (self.land_surface, "noah"),
            "pbl": (self.pbl, "ysu"),
            "microphysics": (self.microphysics, "morrison"),
        }
        for name, (actual, admitted) in required.items():
            if str(actual).lower() != admitted:
                raise ValueError(
                    f"Level-5 built-in adapter admits {name}={admitted!r}, "
                    f"got {actual!r}"
                )
        if str(self.cumulus).lower() not in {*CUMULUS_COMPONENTS, "none"}:
            raise ValueError(
                "Level-5 built-in adapter admits cumulus='gf' (WRF-v4.6.1 "
                "Grell-Freitas, woof/globe/core/kernels/gf.cu), cumulus='ntiedtke' "
                "(WRF-v4.6.1 New Tiedtke, woof/globe/core/kernels/ntiedtke.cu), "
                "cumulus='own' (arwen-massflux-v1, woof/globe/physics/"
                "arwen_massflux.py) or cumulus='none', "
                f"got {self.cumulus!r}"
            )
        if self.gf_ishallow not in {0, 1} or isinstance(self.gf_ishallow, bool):
            raise ValueError("gf_ishallow must be 0 or 1")
        if not isinstance(self.ntiedtke_tiedtke_closure, bool):
            raise ValueError("ntiedtke_tiedtke_closure must be true or false")
        if not isinstance(self.gf_updraft_only_when_downdraft_dry, bool):
            raise ValueError("gf_updraft_only_when_downdraft_dry must be true or false")
        if not isinstance(self.gf_resolved_convergence_closure, bool):
            raise ValueError("gf_resolved_convergence_closure must be true or false")
        if self.radiation_validation_mode not in {"fused", "full"}:
            raise ValueError("radiation_validation_mode must be 'fused' or 'full'")
        if self.sfclay_option not in {1, 91}:
            raise ValueError("sfclay_option must be 1 or 91")
        if self.sfclay_iz0tlnd not in {0, 1, 2, 3} or isinstance(self.sfclay_iz0tlnd, bool):
            raise ValueError("sfclay_iz0tlnd must be 0, 1, 2 or 3")
        if self.ysu_topdown_pblmix not in {0, 1}:
            raise ValueError("ysu_topdown_pblmix must be 0 or 1")
        # The mode table is the CARRIED contract's, and the kernel that
        # switches on it is carried beside it.  Until 2026-09-09 a second
        # refusal stood here for the engine that could not run "fixed" -- its
        # launcher took no flag and its kernel one fewer argument -- and it
        # is retired because that kernel now travels with this package.
        if self.ysu_free_atmosphere_mixing_length not in YSU_FREE_ATMOSPHERE_MIXING_LENGTHS:
            raise ValueError(
                "ysu_free_atmosphere_mixing_length must be one of "
                f"{sorted(YSU_FREE_ATMOSPHERE_MIXING_LENGTHS)}, got "
                f"{self.ysu_free_atmosphere_mixing_length!r}"
            )
        if self.morr_rimed_ice not in {0, 1}:
            raise ValueError("morr_rimed_ice must be 0 or 1")
        if self.noah_thermal_conductivity_option not in {1, 2}:
            raise ValueError("noah_thermal_conductivity_option must be 1 or 2")
        if self.wrf_rrtmg_compatibility not in {"none", "wrf-v4.6.1-v1", "wrf-v4.6.1-v2"}:
            raise ValueError("unknown wrf_rrtmg_compatibility identity")
        for name in (
            "radiation_interval_s", "land_surface_interval_s", "dx_m",
            "water_fix_tolerance_kg_m2", "energy_change_advisory_j_m2",
            "initial_pbl_height_m", "initial_soil_temperature_k",
            "initial_soil_water_fraction", "trace_co2_ppm",
            "stratospheric_floor_k", "stratospheric_floor_pa",
            "stratospheric_relaxation_time_s",
        ):
            raw = getattr(self, name)
            if isinstance(raw, bool) or not math.isfinite(float(raw)):
                raise ValueError(f"{name} must be finite")
        if self.radiation_interval_s <= 0.0 or self.land_surface_interval_s <= 0.0:
            raise ValueError("native physics intervals must be positive")
        if self.stratospheric_floor_pa <= 0.0:
            raise ValueError("stratospheric_floor_pa must be positive")
        if self.stratospheric_relaxation_time_s <= 0.0:
            raise ValueError("stratospheric_relaxation_time_s must be positive")
        if self.radiation_column_chunk < 1:
            raise ValueError("radiation_column_chunk must be positive")
        if (
            isinstance(self.cumulus_column_chunk, bool)
            or int(self.cumulus_column_chunk) != self.cumulus_column_chunk
            or self.cumulus_column_chunk < 1
        ):
            raise ValueError("cumulus_column_chunk must be a positive integer")
        if self.dx_m <= 0.0 or self.water_fix_tolerance_kg_m2 < 0.0:
            raise ValueError("dx_m must be positive and water tolerance nonnegative")
        if self.energy_change_advisory_j_m2 < 0.0:
            raise ValueError("energy_change_advisory_j_m2 must be nonnegative")
        if not 0.0 <= self.initial_soil_water_fraction <= 1.0:
            raise ValueError("initial_soil_water_fraction must lie in [0,1]")
        default_water_tolerance = type(self).__dataclass_fields__[
            "water_fix_tolerance_kg_m2"
        ].default
        if self.water_fix_tolerance_kg_m2 != default_water_tolerance:
            raise ValueError(
                "water_fix_tolerance_kg_m2 is measured and reported, not "
                "enforced: no gate consumes it, so setting "
                f"{self.water_fix_tolerance_kg_m2!r} would change nothing about "
                "when the run refuses.  Read the measured residual from the "
                "receipt tracker maximum_native_water_residual_kg_m2 instead."
            )
        for name in ("land_ice_category", "urban_category"):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or int(value) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("use_monthly_albedo", "read_lai_2d"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be true or false")


__all__ = ["CUMULUS_COMPONENTS", "NativePhysicsOptions"]
