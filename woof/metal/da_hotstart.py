"""Metal pointwise increments for prepared radar hot starts."""
from dataclasses import dataclass
from functools import lru_cache
import platform

import numpy as np

from .array import MetalArray, MetalArrayModule


@dataclass(frozen=True)
class DeviceResult:
    theta: MetalArray
    moisture: MetalArray
    flags: MetalArray


def launch(qv, obs, model, mask, ramp, qsat, cfg, applied=None):
    """Launch with NumPy host preprocessing; return device arrays and flags."""
    if not isinstance(qv, MetalArray) or qv.dtype != np.dtype(np.float32):
        raise TypeError('qv must be a float32 MetalArray')
    shape = qv.shape
    if qv.size == 0 or qv.size > np.iinfo(np.int32).max:
        return None
    ctx = qv.buffer.context
    resident_ramp = isinstance(ramp, MetalArray)
    if resident_ramp and (ramp.buffer.context is not ctx or ramp.shape != shape
                          or ramp.dtype != qv.dtype):
        raise ValueError('pointwise ramp must match resident water vapor')
    host = {'obs': np.asarray(obs, dtype=np.float32),
            'model': np.asarray(model, dtype=np.float32),
            'mask': np.asarray(mask, dtype=np.uint32),
            'qsat': np.asarray(qsat)}
    if not resident_ramp:
        host['ramp'] = np.asarray(ramp)
    if any(value.shape != shape for value in host.values()):
        raise ValueError('pointwise input shapes differ')
    if any(host[name].dtype != qv.dtype for name in ('ramp', 'qsat') if name in host):
        return None
    ledger = applied is not None
    if ledger:
        for key, name in (('thp', 'applied_theta'), ('qv', 'applied_qv')):
            host[name] = np.asarray(applied[key], dtype=np.float32)
            if host[name].shape != shape:
                raise ValueError('pointwise ledger shape differs')
    param_names = {
        'cap': 'max_insertion_dbz', 'echo': 'echo_threshold_dbz',
        'deficit': 'deficit_threshold_db', 'theta_per_db': 'theta_per_db',
        'theta_max': 'max_theta_increment_k', 'theta_min': 'max_theta_decrement_k',
        'rh': 'rh_target', 'qv_max': 'max_qv_increment_kg_kg',
        'qv_min': 'max_qv_decrement_kg_kg', 'clear_obs': 'clear_air_obs_threshold_dbz',
        'clear_model': 'clear_air_model_threshold_dbz',
        'clear_theta_per_db': 'clear_air_theta_per_db', 'clear_rh': 'clear_air_rh_target'}
    values = {key: float(getattr(cfg, name)) for key, name in param_names.items()}
    if any(not np.isfinite(v) or abs(v) > np.finfo(np.float32).max for v in values.values()):
        return None
    params = {key: float(np.float32(v)) for key, v in values.items()}
    params.update(adjust_moisture=int(cfg.adjust_moisture),
                  clear_air=int(cfg.clear_air_enabled), ledger=int(ledger), n=qv.size)
    xp = MetalArrayModule(ctx)
    arrays = {name: xp.asarray(np.ascontiguousarray(value)).view((qv.size,))
              for name, value in host.items()}
    arrays['qv'] = qv.view((qv.size,))
    if resident_ramp:
        arrays['ramp'] = ramp.view((qv.size,))
    if not ledger:
        # The shader only reads these buffers when the ledger is active.
        arrays['applied_theta'] = arrays['applied_qv'] = arrays['qv']
    theta = xp.empty(shape, dtype=np.float32)
    moisture = xp.empty(shape, dtype=np.float32)
    flags = xp.empty(shape, dtype=np.uint32)
    arrays.update(theta=theta.view((qv.size,)), moisture=moisture.view((qv.size,)),
                  flags=flags.view((qv.size,)))
    ctx.submit_kernel('da_hotstart_pointwise', arrays, params, exact_policy=False,
                      grid_size=qv.size, threads_per_group=128)
    return DeviceResult(theta, moisture, flags)


def _height_agl_device(php, phb, shape):
    from woof.core import constants as c

    if not isinstance(php, MetalArray) or not isinstance(phb, MetalArray):
        return None
    shape = tuple(shape)
    if (len(shape) != 3 or any(n <= 0 for n in shape)
            or php.dtype != np.dtype(np.float32) or phb.dtype != php.dtype
            or php.shape != (shape[0] + 1, *shape[1:])
            or phb.shape not in ((shape[0] + 1,), (shape[0] + 1, 1, 1), php.shape)
            or php.buffer.context is not phb.buffer.context
            or php.size > np.iinfo(np.int32).max):
        return None
    ctx = php.buffer.context
    if np.prod(shape) * 4 > min(int(ctx.recommended_working_set_size) // 16,
                                int(ctx.max_buffer_length)):
        return None
    xp = MetalArrayModule(ctx)
    height = xp.empty(shape, dtype=np.float32)
    arrays = {'php': php.view((php.size,)), 'phb': phb.view((phb.size,)),
              'height': height.view((height.size,))}
    params = {'n': height.size, 'plane': shape[1] * shape[2],
              'full_n': php.size, 'base_n': phb.size,
              'base_column': int(phb.size == shape[0] + 1),
              'gravity': float(np.float32(c.G))}
    ctx.submit_kernel('da_hotstart_height', arrays, params, exact_policy=False,
                      grid_size=height.size, threads_per_group=128)
    return height


def _finite_height(height):
    result = height.to_numpy()
    return result if np.isfinite(result).all() else None


def height_agl(php, phb, shape):
    """Return host height from supported resident float32 geopotential layouts."""
    height = _height_agl_device(php, phb, shape)
    return None if height is None else _finite_height(height)


@lru_cache(maxsize=1)
def _numpy_simd_sine():
    if np.__version__ != '2.5.4' or platform.system() != 'Darwin' or platform.machine() != 'arm64':
        return False
    # Sample words across the sine range that distinguish NumPy SIMD from libm.
    angles = [0x3e80be77, 0x3ed7a56e, 0x3f4ac595, 0x3f9779b2,
              0x3fdeafd9, 0x3ffc0e7f, 0x4016f8f3, 0x40308526]
    expected = [0x3e7ec87b, 0x3ed153a4, 0x3f363970, 0x3f6d078a,
                0x3f7c5af1, 0x3f6bf383, 0x3f348555, 0x3ebf8edf]
    values = np.array(angles * 4, dtype=np.uint32).view(np.float32)
    return np.array_equal(np.sin(values).view(np.uint32), expected * 4)


def prepare_ramp(php, phb, shape, cfg):
    """Return host height on fallback, or the host and resident ramp together."""
    height = _height_agl_device(php, phb, shape)
    if height is None:
        return None, None, None
    mode = {'sin2': 0, 'tent': 1, 'uniform': 2}[cfg.ramp_shape]
    with np.errstate(over='ignore', invalid='ignore'):
        base = np.float32(cfg.warm_base_height_m)
        top = np.float32(cfg.warm_top_height_m)
        width = top - base
    if (not np.isfinite(width) or width <= 0
            or (mode == 0 and not _numpy_simd_sine())):
        return _finite_height(height), None, None
    ctx = height.buffer.context
    xp = MetalArrayModule(ctx)
    ramp = xp.empty(height.shape, dtype=np.float32)
    flags = xp.empty(height.shape, dtype=np.uint32)
    ctx.submit_kernel('da_hotstart_ramp',
                      {name: array.view((height.size,)) for name, array in
                       (('height', height), ('ramp', ramp), ('flags', flags))},
                      dict(base=float(base), top=float(top), shape=mode, n=height.size),
                      exact_policy=False, grid_size=height.size, threads_per_group=128)
    if np.any(flags.to_numpy()):
        return _finite_height(height), None, None
    return None, ramp.to_numpy(), ramp


class MetalHotstartPointwise:
    """One member's optional pointwise accelerator; host diagnostics stay shared."""
    def __init__(self, qv, device_ramp=None):
        self.qv = qv
        self.device_ramp = device_ramp
        self.used = False

    def __call__(self, obs, model, valid, ramp, qsat, cfg, applied):
        result = launch(self.qv, obs, model, valid,
                        ramp if self.device_ramp is None else self.device_ramp,
                        qsat, cfg, applied)
        if result is None:
            return None
        flags = result.flags.to_numpy()
        if np.any(flags & np.uint32(4)):
            return None
        self.used = True
        return (result.theta.to_numpy(), result.moisture.to_numpy(),
                (flags & np.uint32(1)) != 0, (flags & np.uint32(2)) != 0)
