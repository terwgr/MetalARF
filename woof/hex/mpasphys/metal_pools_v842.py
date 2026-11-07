"""Resident Metal views of hydrated MPAS v8.4.2 Registry pools.

Registry hydration and native time levels stay owned by NativePoolHydration
and FrameworkPhysics. These views pack an explicit solve or full extent and
publish only at caller-requested boundaries.
"""
from __future__ import annotations
import numpy as np
from woof.metal.array import MetalArrayModule


class MetalPoolViews:
    def __init__(self, native, layout, context):
        self.native, self.layout, self.arrays = native, layout, MetalArrayModule(context)
        self.buffers = {}
        self.prepared_buffers = {}
        self.rows = {(r['pool'], alias): r for r in layout.fields
                     for alias in (r['name'], r['pool_name'])}

    def field(self, pool, name, *, level=1, full=False):
        row = self.rows[pool, name]
        key = pool, row['pool_name'], level
        entity = next((d for d in row['dimensions'] if d in ('nCells', 'nEdges', 'nVertices')), None)
        selection = (..., slice(0, self.layout.dimensions[entity + 'Solve'])) if entity and not full else (...,)
        if key in self.buffers:
            host, device, old_selection = self.buffers[key]
            if host[old_selection].shape != host[selection].shape:
                raise ValueError(f'cached extent conflict for {pool}.{name} level {level}')
            return device
        storage = getattr(self.native, 'field_storage', None)
        host = storage(pool, row['pool_name'], level=level) if storage else self.native.get_field(pool, row['pool_name'], level=level)
        packed = np.ascontiguousarray(host[selection]).reshape(host[selection].shape)
        device = self.arrays.asarray(packed)
        self.buffers[key] = host, device, selection
        return device

    def prepared(self, name):
        """Pack an allocated native i[,k],one-j variable as [k,cell]."""
        if name in self.prepared_buffers:
            return self.prepared_buffers[name]
        value = self.native.variable_storage(name)
        if value is None:
            raise RuntimeError(f'native prepared array {name} is not allocated')
        host, lower = value
        view = self._prepared_view(name, host, lower)
        self.prepared_buffers[name] = self.arrays.asarray(np.ascontiguousarray(view))
        return self.prepared_buffers[name]

    def _prepared_view(self, name, host, lower):
        if host.ndim not in (2,3,4) or host.shape[2 if host.ndim==4 else -1] != 1 or lower[0] != 1:
            raise ValueError(f'native prepared {name} has unsupported bounds')
        count = self.layout.dimensions['nCellsSolve']
        view = host[:count,:,0,:].transpose(2,1,0) if host.ndim==4 else host[:count,:,0].T if host.ndim==3 else host[:count,0]
        return view

    def refresh(self):
        """Refresh cached Registry views after explicit host initialization."""
        for (pool, name, level), (host, device, selection) in self.buffers.items():
            host[...] = self.native.get_field(pool, name, level=level)
            device[...] = np.ascontiguousarray(host[selection])

    def publish(self):
        """Publish resident words into their original native time levels."""
        for (pool, name, level), (host, device, selection) in self.buffers.items():
            host[selection] = device.to_numpy()
            self.native.put_field(pool, name, host, level=level,
                                  nlevels=self.native._fields[pool, name][2])
        for name, device in self.prepared_buffers.items():
            host, lower = self.native.variable_storage(name)
            self._prepared_view(name, host, lower)[...] = device.to_numpy()

    def prepared_restart_state(self):
        """Save resident module arrays, including arrays not yet viewed on Metal."""
        if not self.native._initialized:
            raise RuntimeError('prepared state requires initialized MPAS physics')
        self.publish()
        return {name: (value.copy(order='F'), lower)
                for name, (value, lower) in self.native._variables.items() if value.ndim >= 2}

    def restore_prepared_restart_state(self, state):
        for name, (value, lower) in state.items():
            current = self.native.variable_storage(name)
            if current is None or current[0].shape != value.shape or current[0].dtype != value.dtype or current[1] != lower:
                raise ValueError(f'native prepared restart layout differs at {name}')
        for name, (value, _) in state.items():
            self.native.variable_storage(name)[0][...] = value
        for name, device in self.prepared_buffers.items():
            if name not in state:
                raise ValueError(f'native prepared restart lacks {name}')
            host, lower = self.native.variable_storage(name)
            device[...] = np.ascontiguousarray(self._prepared_view(name, host, lower))

    def snapshot(self):
        """Copy every native Registry numeric field and time level, including dummy cells.

        This is a pool snapshot, not a complete scheme or forecast restart.
        """
        self.publish()
        return {f'{pool}.{name}.{level}': self.native.get_field(pool, name, level=level)
                for (pool, name), (_, _, levels) in self.native._fields.items()
                for level in range(1, levels + 1)}

    def restore(self, fields):
        """Validate the entire pool snapshot before restoring native and resident words."""
        expected = {f'{pool}.{name}.{level}'
                    for (pool, name), (_, _, levels) in self.native._fields.items()
                    for level in range(1, levels + 1)}
        if set(fields) != expected:
            raise ValueError('native pool snapshot field set differs')
        for key, value in fields.items():
            pool, name, level = key.rsplit('.', 2)
            dtype, shape, _ = self.native._fields[pool, name]
            if value.shape != shape or value.dtype != dtype:
                raise ValueError(f'native pool snapshot layout differs at {key}')
        for key, value in fields.items():
            pool, name, level_text = key.rsplit('.', 2)
            level = int(level_text)
            self.native.put_field(pool, name, value, level=level,
                                  nlevels=self.native._fields[pool, name][2])
            cached = self.buffers.get((pool, name, level))
            if cached is not None:
                host, device, selection = cached
                host[...] = value
                device[...] = np.ascontiguousarray(value[selection]).reshape(device.shape)
