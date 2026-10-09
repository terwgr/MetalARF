"""The ATMS channel set: passbands, polarization, noise.

Source: the JPSS ATMS instrument description (NOAA/NESDIS STAR ATMS SDR
user's guide; the channel plan is the one every ATMS SDR product
carries).  Centre frequencies in GHz, per-sideband bandwidths in MHz,
the specified NEdT in kelvin at 300 K.  Channels 6 and 11 to 15 are
double or quadruple sideband: the passband list holds every sub-band
centre and the radiance the operator returns is the mean over the
sub-bands, each integrated across its own width.

Polarization is "quasi": the ATMS reflector rotates, so the sensed
polarization turns with the scan angle.  A QV channel measures the
vertically polarized surface at nadir and a mix of V and H away from it,

    T(QV) = T_v cos^2(a) + T_h sin^2(a),    T(QH) = T_h cos^2(a) + T_v sin^2(a),

with ``a`` the scan angle from nadir.  The forward operator applies this
mix to the surface emissivity; the atmosphere is unpolarized.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The oxygen line the upper sounding channels sit on (GHz).
F0_GHZ = 57.290344


@dataclass(frozen=True)
class Channel:
    number: int
    passbands_ghz: tuple[float, ...]
    bandwidth_mhz: float
    polarization: str  # "QV" | "QH"
    nedt_k: float
    beamwidth_deg: float
    role: str

    @property
    def centre_ghz(self) -> float:
        return sum(self.passbands_ghz) / len(self.passbands_ghz)

    @property
    def label(self) -> str:
        return f"ch{self.number:02d}"


def _sidebands(centre: float, *offsets: float) -> tuple[float, ...]:
    bands = [centre]
    for offset in offsets:
        bands = [b + s * offset for b in bands for s in (-1.0, 1.0)]
    return tuple(sorted(bands))


CHANNELS: tuple[Channel, ...] = (
    Channel(1, (23.8,), 270.0, "QV", 0.5, 5.2, "window / water vapour"),
    Channel(2, (31.4,), 180.0, "QV", 0.6, 5.2, "window"),
    Channel(3, (50.3,), 180.0, "QH", 0.7, 2.2, "window / surface"),
    Channel(4, (51.76,), 400.0, "QH", 0.5, 2.2, "temperature, lower troposphere"),
    Channel(5, (52.8,), 400.0, "QH", 0.5, 2.2, "temperature, lower troposphere"),
    Channel(6, _sidebands(53.596, 0.115), 170.0, "QH", 0.5, 2.2, "temperature, mid troposphere"),
    Channel(7, (54.40,), 400.0, "QH", 0.5, 2.2, "temperature, mid troposphere"),
    Channel(8, (54.94,), 400.0, "QH", 0.5, 2.2, "temperature, upper troposphere"),
    Channel(9, (55.50,), 330.0, "QH", 0.5, 2.2, "temperature, tropopause"),
    Channel(10, (F0_GHZ,), 330.0, "QH", 0.75, 2.2, "temperature, lower stratosphere"),
    Channel(11, _sidebands(F0_GHZ, 0.217), 78.0, "QH", 1.2, 2.2, "temperature, stratosphere"),
    Channel(12, _sidebands(F0_GHZ, 0.3222, 0.048), 36.0, "QH", 1.2, 2.2, "temperature, stratosphere"),
    Channel(13, _sidebands(F0_GHZ, 0.3222, 0.022), 16.0, "QH", 1.5, 2.2, "temperature, stratosphere"),
    Channel(14, _sidebands(F0_GHZ, 0.3222, 0.010), 8.0, "QH", 2.4, 2.2, "temperature, upper stratosphere"),
    Channel(15, _sidebands(F0_GHZ, 0.3222, 0.0045), 3.0, "QH", 3.6, 2.2, "temperature, upper stratosphere"),
    Channel(16, (88.2,), 2000.0, "QV", 0.3, 2.2, "window"),
    Channel(17, (165.5,), 3000.0, "QH", 0.6, 1.1, "window / water vapour"),
    Channel(18, _sidebands(183.31, 7.0), 2000.0, "QH", 0.8, 1.1, "water vapour"),
    Channel(19, _sidebands(183.31, 4.5), 2000.0, "QH", 0.8, 1.1, "water vapour"),
    Channel(20, _sidebands(183.31, 3.0), 1000.0, "QH", 0.8, 1.1, "water vapour"),
    Channel(21, _sidebands(183.31, 1.8), 1000.0, "QH", 0.8, 1.1, "water vapour"),
    Channel(22, _sidebands(183.31, 1.0), 500.0, "QH", 0.9, 1.1, "water vapour"),
)

#: The temperature-sounding set this leg is graded on: the oxygen-band
#: channels whose weighting functions peak from the lower troposphere
#: (channel 4) to the upper stratosphere (channel 15).  Channel 3 is a
#: window channel at 50.3 GHz and is scored as the surface-sensitive
#: control, not as a sounding channel.
TEMPERATURE_SOUNDING_CHANNELS: tuple[int, ...] = tuple(range(4, 16))

#: Scan geometry: 96 beams, 1.11 degree steps, symmetric about nadir.
FOV_COUNT = 96
SCAN_STEP_DEG = 1.11


def scan_angle_deg(fov_index) -> float:
    """Scan angle from nadir of beam ``fov_index`` (0-based), signed."""
    return (fov_index - (FOV_COUNT - 1) / 2.0) * SCAN_STEP_DEG


def channel(number: int) -> Channel:
    for entry in CHANNELS:
        if entry.number == number:
            return entry
    raise KeyError(f"ATMS has no channel {number}")
