"""In-situ instrumentation for WOOF global: the ledger that watches the
running model every step (budgets, spectra, physics component tendencies,
tripwires) and snapshots the state at onset.

Diagnostic only: nothing here enters the config hash or changes a
prognostic bit.  Model-side hooks are one observer slot on the model
(``MoistHybridModel.observer``) and one on each physics suite; every
instrument lives in this package.
"""
from .budgets import CONSERVED_TERMS, LEDGER_NAMES, LEDGER_TERMS, ledger_row
from .capture import CAPTURE_NAMES, ComponentCapture, attach_capture
from .energy import ENERGY_COMPONENTS, STEP_OPERATORS, OperatorEnergyLedger
from .ledger import INSITU_SCHEMA, LEDGER_NAME, InsituLedger, insitu_owned_files
from .options import InsituOptions, insitu_options_from_table
from .spectra import SpectralKineticEnergy
from .tripwires import THRESHOLDS, TripwireSet, tripwire_specs

__all__ = [
    "CAPTURE_NAMES",
    "CONSERVED_TERMS",
    "ComponentCapture",
    "ENERGY_COMPONENTS",
    "INSITU_SCHEMA",
    "InsituLedger",
    "InsituOptions",
    "LEDGER_NAME",
    "LEDGER_NAMES",
    "LEDGER_TERMS",
    "OperatorEnergyLedger",
    "STEP_OPERATORS",
    "SpectralKineticEnergy",
    "THRESHOLDS",
    "TripwireSet",
    "attach_capture",
    "insitu_options_from_table",
    "insitu_owned_files",
    "ledger_row",
    "tripwire_specs",
]
