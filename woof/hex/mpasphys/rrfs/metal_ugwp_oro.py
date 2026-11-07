"""Resident RRFS orographic UGWP slot for Metal."""

import numpy as np

from woof.metal.array import MetalArray
from .. import ugwp_oro_v842 as oro


class MetalMpasUgwpOro:
    """UGWP has no private restart state; native pools own its fields."""

    def restart_state(self):
        return {}

    def restore_restart_state(self, state):
        if state != {}:
            raise ValueError('UGWP physical state belongs to native pools')

    def launch_slot(self, call):
        if (call.slot, call.scheme) != ('gwdo', oro.SCHEME):
            raise ValueError('UGWP service requires the orographic gravity-wave slot')
        options = call.options
        if options['config_ngw_scheme']:
            raise NotImplementedError('Metal UGWP does not support nonstationary gravity waves')
        diags = options['config_ugwp_diags']
        if not isinstance(diags, (bool, np.bool_)):
            raise ValueError('UGWP diagnostic flag must be boolean')
        pools, dt = call.pools, np.float32(call.dt)
        nz, nc = int(pools['nVertLevels']), int(pools['nCellsSolve'])
        if nz < 2 or nc < 1 or nz * nc > np.iinfo(np.int32).max:
            raise ValueError('Metal UGWP requires at least two levels and a supported solve extent')
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('UGWP requires a positive finite timestep')
        native, xp = pools.native, pools.cp
        prep, tend, diag, sfc, mesh = (pools[name] for name in
            ('prepared', 'tend_physics', 'diag_physics', 'sfc_input', 'mesh'))
        raw = pools['constants']
        constants = {name: np.float32(raw[native_name]) for name, native_name in
                     (('g', 'gravity'), ('cp', 'cp'), ('rd', 'R_d'),
                      ('rv', 'R_v'), ('fv', 'ep_1'), ('pi_c', 'pii'))}
        length = np.float32(native.get_real_config('config_len_disp'))
        if not np.isfinite(length) or length <= 0 or not all(np.isfinite(v) for v in constants.values()):
            raise ValueError('UGWP requires finite native constants and positive resolved grid length')

        ctx = None
        def require(name, value, dtype, shape):
            nonlocal ctx
            if (not isinstance(value, MetalArray) or value.dtype != np.dtype(dtype)
                    or value.shape != shape or not value.flags.c_contiguous):
                raise ValueError(f'UGWP {name} requires contiguous {np.dtype(dtype)} {shape}')
            if ctx is None:
                ctx = value.buffer.context
            elif value.buffer.context is not ctx:
                raise ValueError('UGWP arrays belong to different Metal contexts')
            return value

        dynamic = dict(uproj='u_p', vproj='v_p', t1='t_p', q1='qv_p', prsl='pres_hydd_p',
                       prslk='pi_p', zl='zmid_p', dz='dz_p')
        columns = {key: prep[name][:nz] for key, name in dynamic.items()}
        columns.update(rublten=tend['rublten'], rvblten=tend['rvblten'],
                       hpbl=diag['hpbl'], br1=diag['br'], xland1=sfc['xland'])
        for scale, suffix in (('ls', ''), ('ss', 'ss')):
            columns['var' + suffix], columns['oc1' + suffix] = sfc['var2d' + scale], sfc['con' + scale]
            for direction in range(1, 5):
                for statistic in ('oa', 'ol'):
                    columns[f'{statistic}2d{direction}{suffix}'] = sfc[f'{statistic}{direction}{scale}']
        for name in oro.LAYER_INPUTS:
            require(name, columns[name], np.float32, (nz, nc))
        for name in oro.SURFACE_INPUTS:
            if name not in ('sina', 'cosa', 'dxmeter'):
                require(name, columns[name], np.float32, (nc,))
        prsi = require('prsi', prep['pres2_hydd_p'][:nz + 1], np.float32, (nz + 1, nc))
        kpbl = require('kpbl', diag['kpbl'], np.int32, (nc,))
        density = require('meshDensity', mesh['meshDensity'], np.float32, (nc,))
        require('rthblten', tend['rthblten'], np.float32, (nz, nc))

        layer_names = oro.LAYER_OUTPUTS[:12 if diags else 4]
        surface_names = oro.SURFACE_OUTPUTS[:10 if diags else 2]
        output_names = (*layer_names, *surface_names, 'rubldiff', 'rvbldiff', 'rthblten')
        targets = {}
        for name in output_names:
            target = tend[name] if name in ('rublten', 'rvblten', 'rthblten') else diag[name]
            shape = (nz, nc) if name in (*layer_names, 'rubldiff', 'rvbldiff', 'rthblten') else (nc,)
            targets[name] = require(name, target, np.float32, shape)
        prepared_targets = {}
        for name in output_names:
            if name in ('rubldiff', 'rvbldiff'):
                continue
            value = prep[name + '_p']
            expected = (nc,) if name in surface_names else (nz, nc)
            prepared_targets[name] = require(name + '_p', value if len(expected) == 1 else value[:nz], np.float32, expected)
        from_names = ('rublten', 'rvblten', 'rthblten', 'kpbl', 'hpbl', 'br', 'xland')
        from_names += tuple(f'{name}{scale}' for scale in ('ls', 'ss')
                            for name in ('var2d', 'con', 'oa1', 'oa2', 'oa3', 'oa4', 'ol1', 'ol2', 'ol3', 'ol4'))
        from_targets = {}
        for name in from_names:
            source = (tend[name] if name in ('rublten', 'rvblten', 'rthblten') else
                      diag[name] if name in ('kpbl', 'hpbl', 'br') else sfc[name])
            dtype = np.int32 if name == 'kpbl' else np.float32
            shape = (nz, nc) if name in ('rublten', 'rvblten', 'rthblten') else (nc,)
            require(name, source, dtype, shape)
            target = prep[name + '_p']
            from_targets[name] = (source, require(name + '_p', target[:nz] if len(shape) == 2 else target, dtype, shape))
        orientation = {name: require(name + '_p', prep[name + '_p'], np.float32, (nc,))
                       for name in ('sina', 'cosa', 'dx')}
        sizes = (10*nz*nc, 26*nc, 12*nz*nc, 10*nc, 2*nz*nc, nc*16*(nz+2), 1)
        if max(sizes) * 4 > ctx.max_buffer_length or sum(sizes) * 4 > ctx.recommended_working_set_size:
            raise MemoryError('UGWP workspace exceeds the Metal working set')
        layers, surface = xp.empty((10, nz, nc), np.float32), xp.empty((26, nc), np.float32)
        out, sout = xp.empty((12, nz, nc), np.float32), xp.empty((10, nc), np.float32)
        diff, status = xp.empty((2, nz, nc), np.float32), xp.zeros((1,), np.int32)
        workspace = xp.empty((nc, 16, nz + 2), np.uint32)
        for i, name in enumerate(oro.LAYER_INPUTS):
            layers[i][...] = columns[name]
        for i, name in enumerate(oro.SURFACE_INPUTS):
            if name == 'sina': surface[i].fill(0)
            elif name == 'cosa': surface[i].fill(1)
            elif name != 'dxmeter': surface[i][...] = columns[name]
        ctx.submit_kernel('hexrrfs_ugwp_oro_dx', dict(density=density, dx=surface[5]),
                          dict(length=length, n=np.int32(nc)), exact_policy=True, grid_size=nc)
        ctx.submit_kernel('hexrrfs_ugwp_oro_run',
                          dict(layers=layers, prsi0=prsi, surface=surface, kpbl0=kpbl, out=out, sout=sout, status=status, workspace=workspace),
                          dict(constants, dt=dt, nz=np.int32(nz), nc=np.int32(nc), diags=np.int32(diags)), exact_policy=True, grid_size=nc)
        if status.to_numpy()[0]:
            raise ValueError('UGWP_INVALID_KPBL: native small-scale path would read beyond layer storage')
        for i, name in enumerate(('rublten', 'rvblten')):
            ctx.submit_kernel('hexrrfs_ugwp_oro_diff',
                              dict(before=columns[name].view((nz*nc,)), after=out[i].view((nz*nc,)), diff=diff[i].view((nz*nc,))),
                              dict(n=np.int32(nz*nc)), exact_policy=True, grid_size=nz*nc)
        outputs = {name: out[i] for i, name in enumerate(layer_names)}
        outputs.update({name: sout[i] for i, name in enumerate(surface_names)})
        outputs.update(rubldiff=diff[0], rvbldiff=diff[1], rthblten=tend['rthblten'])
        for name, (source, target) in from_targets.items():
            if name not in prepared_targets:
                target[...] = source
        for name, value in zip(('sina', 'cosa', 'dx'), (surface[0], surface[1], surface[5])):
            orientation[name][...] = value
        for name, target in targets.items():
            if name != 'rthblten': target[...] = outputs[name]
        for name, target in prepared_targets.items():
            target[...] = outputs[name]
        return outputs
