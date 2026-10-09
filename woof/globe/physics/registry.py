"""Fail-closed registry for physics adapters admitted to WOOF global.

The registry is dependency-light: importing it does not import CuPy, model
kernels, or the global dycore.  Adapter option validation is a pure function
and therefore runs during TOML loading, before transform construction or any
device access.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Callable


_SHA_ZERO = "0" * 64
_ALLOWED_STATUS = {"device-pending", "experimental", "validated"}


@dataclass(frozen=True)
class GlobalPhysicsAdapterRegistration:
    name: str
    factory: Callable
    contract: dict[str, object]
    options_validator: Callable[[dict[str, object]], dict[str, object]] | None = None
    #: Maps the NORMALIZED options (the validator's output, what the adapter
    #: is built from) to the payload that names their trajectory (what the
    #: config hash and the receipt carry).  None: the normalized options are
    #: their own identity.  Kept apart from the validator because an option
    #: that changes no arithmetic in some configurations (a scheme-scoped
    #: flag) or that spells the arithmetic every earlier checkpoint was
    #: written under must stay out of the hash there, while the adapter
    #: must still be built from every normalized option: built from the
    #: identity payload instead, it read the dropped option as its default,
    #: a different arithmetic wherever the default is not the value dropped
    #: (measured 2026-09-05: a config spelling YSU's "wrf-layer" length ran
    #: the "fixed" length, and the probe that restarted the control read a
    #: kernel word that was not the control's).
    options_identity: Callable[[dict[str, object]], dict[str, object]] | None = None

    @property
    def contract_hash(self) -> str:
        raw = json.dumps(
            self.contract, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    def validate_options(self, options: dict[str, object]) -> dict[str, object]:
        if not isinstance(options, dict):
            raise TypeError("global physics adapter options must be a dict")
        if self.options_validator is None:
            return dict(options)
        normalized = self.options_validator(dict(options))
        if not isinstance(normalized, dict):
            raise TypeError(
                f"adapter {self.name!r} options validator must return a dict"
            )
        # Prove the normalized identity is JSON serializable now, not when a
        # checkpoint or receipt is emitted after device work.
        json.dumps(normalized, sort_keys=True, allow_nan=False)
        return normalized

    def identity_of(self, options: dict[str, object]) -> dict[str, object]:
        """The hash payload of NORMALIZED ``options`` (see ``options_identity``)."""
        if not isinstance(options, dict):
            raise TypeError("global physics adapter options must be a dict")
        if self.options_identity is None:
            return dict(options)
        identity = self.options_identity(dict(options))
        if not isinstance(identity, dict):
            raise TypeError(
                f"adapter {self.name!r} options identity must return a dict"
            )
        json.dumps(identity, sort_keys=True, allow_nan=False)
        return identity


_REGISTRY: dict[str, GlobalPhysicsAdapterRegistration] = {}


def _digest(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"adapter {name} must be a SHA-256 hex digest")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"adapter {name} must be hexadecimal") from exc
    return value.lower()


def _normalize_contract(contract: dict[str, object]) -> dict[str, object]:
    if not isinstance(contract, dict):
        raise TypeError("global physics adapter contract must be a dict")
    value = dict(contract)
    # Third-party v1 registrations are retained as experimental candidates;
    # absence of device evidence can never be interpreted as validation.
    if value.get("schema") == "gpuwm.arwen-global-native-physics-adapter/v1":
        value.update(
            schema="gpuwm.arwen-global-native-physics-adapter/v2",
            admission_status="experimental",
            device_evidence_sha256=_SHA_ZERO,
            limitations=[
                "legacy v1 adapter contract: no Level-5 device qualification attached"
            ],
        )
    required = {
        "schema", "scheme_identity", "backend", "precision",
        "required_fields", "pressure_convention", "vertical_coordinate",
        "surface_state", "restart_contract", "budget_contract",
        "evidence_receipt_sha256", "arithmetic_sha256",
        "admission_status", "device_evidence_sha256", "limitations",
    }
    if set(value) != required:
        raise ValueError(
            f"adapter contract keys {sorted(value)} do not match {sorted(required)}"
        )
    if value["schema"] != "gpuwm.arwen-global-native-physics-adapter/v2":
        raise ValueError("adapter contract schema must be v2")
    if not isinstance(value["scheme_identity"], dict) or not value["scheme_identity"]:
        raise ValueError("adapter scheme_identity must be a nonempty dict")
    if not isinstance(value["required_fields"], list) or not value["required_fields"]:
        raise ValueError("adapter required_fields must be a nonempty list")
    if not all(isinstance(item, str) and item for item in value["required_fields"]):
        raise ValueError("adapter required_fields entries must be nonempty strings")
    if value["admission_status"] not in _ALLOWED_STATUS:
        raise ValueError(
            f"adapter admission_status must be one of {sorted(_ALLOWED_STATUS)}"
        )
    if not isinstance(value["limitations"], list) or not all(
        isinstance(item, str) and item for item in value["limitations"]
    ):
        raise ValueError("adapter limitations must be a list of nonempty strings")
    for digest_name in (
        "evidence_receipt_sha256", "arithmetic_sha256", "device_evidence_sha256"
    ):
        value[digest_name] = _digest(value[digest_name], digest_name)
    if value["admission_status"] == "validated" and value["device_evidence_sha256"] == _SHA_ZERO:
        raise ValueError("validated adapter requires nonzero device evidence")
    json.dumps(value, sort_keys=True, allow_nan=False)
    return value


def register_global_physics_adapter(
    name: str,
    factory: Callable,
    contract: dict[str, object],
    *,
    options_validator: Callable[[dict[str, object]], dict[str, object]] | None = None,
    options_identity: Callable[[dict[str, object]], dict[str, object]] | None = None,
    replace: bool = False,
) -> GlobalPhysicsAdapterRegistration:
    key = str(name).strip().lower()
    if not key or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in key):
        raise ValueError("global physics adapter name must be nonempty lowercase slug text")
    if not callable(factory):
        raise TypeError("global physics adapter factory must be callable")
    if options_validator is not None and not callable(options_validator):
        raise TypeError("global physics adapter options_validator must be callable")
    if options_identity is not None and not callable(options_identity):
        raise TypeError("global physics adapter options_identity must be callable")
    normalized = _normalize_contract(contract)
    if key in _REGISTRY and not replace:
        raise ValueError(f"global physics adapter {key!r} is already registered")
    registration = GlobalPhysicsAdapterRegistration(
        key, factory, normalized, options_validator, options_identity
    )
    _REGISTRY[key] = registration
    return registration


def get_global_physics_adapter(name: str) -> GlobalPhysicsAdapterRegistration:
    key = str(name).strip().lower()
    try:
        return _REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(_REGISTRY)) or "none"
        raise ValueError(
            f"WOOF global native physics adapter {key!r} is not admitted; "
            f"registered adapters: {available}"
        ) from exc


def validate_global_physics_options(
    name: str, options: dict[str, object]
) -> dict[str, object]:
    return get_global_physics_adapter(name).validate_options(options)


def identity_of_global_physics_options(
    name: str, options: dict[str, object]
) -> dict[str, object]:
    """The hash payload of adapter ``name``'s NORMALIZED ``options``."""
    return get_global_physics_adapter(name).identity_of(options)


def registered_global_physics_adapters() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


def global_physics_manifest() -> dict[str, object]:
    return {
        name: {
            "admission_status": registration.contract["admission_status"],
            "contract": registration.contract,
            "contract_hash": registration.contract_hash,
            "options_validation": (
                "registered-pure-validator"
                if registration.options_validator is not None
                else "identity-only"
            ),
        }
        for name, registration in sorted(_REGISTRY.items())
    }


__all__ = [
    "GlobalPhysicsAdapterRegistration",
    "get_global_physics_adapter",
    "identity_of_global_physics_options",
    "global_physics_manifest",
    "register_global_physics_adapter",
    "registered_global_physics_adapters",
    "validate_global_physics_options",
]
