"""Physics implementations and adapter contracts for WOOF global."""
from .arwen_bridge import NativeArwenPhysicsBridge
from .builtin_adapters import ensure_builtin_global_physics_adapters
from .exchange import PhysicsExchange, PhysicsResult
from .native_options import NativePhysicsOptions
from .native_suite import ArwenCudaColumnSuite
from .reference import ReferencePhysics, ReferencePhysicsOptions

__all__ = [
    "ArwenCudaColumnSuite",
    "NativeArwenPhysicsBridge",
    "NativePhysicsOptions",
    "PhysicsExchange",
    "PhysicsResult",
    "ReferencePhysics",
    "ReferencePhysicsOptions",
    "ensure_builtin_global_physics_adapters",
]
