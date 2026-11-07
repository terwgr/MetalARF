"""RRFS RUC land and sea-ice slots on the resident Metal pools."""

import numpy as np

from woof.metal.array import MetalArray

from . import metal_ruc_boundaries as boundaries
from . import ruc_lsm_8c9f1a8 as ruc
from .metal_ruc_ice import solve_ice
from .metal_ruc_land import solve_land
from .metal_ruc_thermodynamics import saturation_table


class MetalRucSlots:
    """Prepare independent native work arrays and publish the driver outputs."""

    def restart_state(self):
        # The native pools and startup's immutable tables own the physical state.
        return {}

    def restore_restart_state(self, payload):
        if payload != {}:
            raise ValueError('RRFS RUC physical state belongs to native pools')
        self._rrfs_ruc_constants = None

    def launch_slot(self, call):
        if call.scheme != ruc.SCHEME or call.slot not in ('land_surface', 'seaice'):
            raise ValueError('RRFS RUC requires its land or sea-ice slot')
        pools = call.pools
        native, xp = pools.native, pools.cp
        if getattr(self, 'native', None) is not native:
            self.native = native
            self._rrfs_ruc_constants = None
        entry = 'land_ruc_land' if call.slot == 'land_surface' else 'ice_ruc_ice'
        ice = entry == 'ice_ruc_ice'
        options = dict(pools.session.layout.configs)
        options.update(call.options)
        landuse = native.get_text_field('sfc_input', 'mminlu')
        scalars = dict(myjpbl=0, frpcpn=1, rdlai2d=1,
                       restart=options['config_do_restart'], cycling=options['config_do_DAcycling'],
                       mminlu={'USGS': 1, 'MODIS': 2, 'MODIFIED_IGBP_MODIS_NOAH': 3}[landuse])
        if not ice:
            scalars['lakemodel'] = options['config_lakemodel']
        for field in ruc.ARGUMENTS[entry]:
            name = field['name']
            if field['dim'] or name in scalars or name in ('spp_lsm', 'mosaic_lu', 'mosaic_soil', 'dt', 'ktau'):
                continue
            if name in ('isice', 'iswater'):
                value = pools['sfc_input'][name]
                scalars[name] = value.to_numpy().item() if isinstance(value, MetalArray) else np.asarray(value).item()
            else:
                scalars[name] = ruc._native_scalar(self, name)
        interval = ruc._native_interval(self)
        dt = call.dt if interval is None else interval
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('RRFS RUC requires the initialized positive PBL interval')
        scalars.update(dt=dt, ktau=call.itimestep)
        nc = int(pools['nCellsSolve'])
        if not nc:
            return
        sfc, diag, prepared = pools['sfc_input'], pools['diag_physics'], pools['prepared']
        arrays = {}
        for field in ruc.ARGUMENTS[entry]:
            name = field['name']
            if not field['dim']:
                if name in ('spp_lsm', 'mosaic_lu', 'mosaic_soil'):
                    flag = diag['spp_lsm_loc' if name == 'spp_lsm' else name]
                    if not isinstance(flag, MetalArray) or flag.dtype != np.int32:
                        raise ValueError(f'RRFS RUC {name} requires a resident int32 flag')
                    if flag.size == 1 and nc != 1:
                        arrays[name] = xp.empty((nc,), np.int32)
                        arrays[name].fill(flag.to_numpy().item())
                    elif flag.size == nc:
                        arrays[name] = flag.view((nc,)).copy()
                    else:
                        raise ValueError(f'RRFS RUC {name} must match solve columns')
                continue
            if name in ('pattern_spp_lsm', 'field_sf'):
                arrays[name] = xp.zeros((int(scalars['kme'] - scalars['kms'] + 1), nc), np.float32)
                continue
            if name == 'lakemask':
                arrays[name] = xp.zeros((nc,), np.float32)
                continue
            if name in ('rainbl', 'frzfrac'):
                arrays[name] = xp.empty((nc,), np.float32)
                continue
            if name in ruc.ATM_BINDINGS:
                source = prepared[ruc.ATM_BINDINGS[name]]
            elif name == 'globalcells':
                source = pools['mesh']['indexToCellID']
            elif name in ruc.SFC_BINDINGS:
                source = sfc[ruc.SFC_BINDINGS[name]]
            else:
                source = diag['qcg' if not ice and name == 'qsg' else ruc.DIAG_BINDINGS.get(name, name)]
            if not isinstance(source, MetalArray) or not source.shape or source.shape[-1] != nc:
                raise ValueError(f'RRFS RUC {name} requires compact solve-cell storage')
            arrays[name] = source.copy()
        boundaries.broadcast_soil_profile(sfc['zs'], arrays['zs'], solve_cells=nc)
        micro = options['config_microp_scheme'].strip() != 'off'
        convection = options['config_convection_scheme'].strip() != 'off'
        forcing = {'gsw': diag['gsw'], 'sfc_albedo': diag['sfc_albedo']}
        if micro:
            forcing.update(sr=diag['sr'], rainncv=diag['rainncv'])
        if convection:
            forcing['raincv'] = diag['raincv']
        boundaries.prepare_forcing(forcing, {'sr_p': arrays['frzfrac'], 'rainbl_p': arrays['rainbl'],
                                              'swdown_p': xp.empty((nc,), np.float32)},
                                   solve_cells=nc, microphysics=micro, convection=convection)
        threshold = scalars['xice_threshold']
        if ice:
            boundaries.prepare_seaice_background(arrays['xice'], sfc['sfc_albbck'], arrays['albbck'],
                                                  solve_cells=nc, threshold=threshold)
            boundaries.prepare_seaice({'xice': arrays['xice'], 'sfc_albedo': diag['sfc_albedo'],
                                       'sfc_emiss': diag['sfc_emiss']},
                                      {'sfc_albedo_p': arrays['alb'], 'sfc_emiss_p': arrays['emiss']},
                                      solve_cells=nc, threshold=threshold)
            ice_skin = xp.empty((nc,), np.float32)
            boundaries.correct_seaice_skin(arrays['xice'], arrays['soilt'], prepared['tsk_sea'], ice_skin,
                                            solve_cells=nc, threshold=threshold)
            arrays['soilt'][...] = ice_skin
            for name in ruc.ICE_FINISH_FIELDS:
                if name != 'tsk' and name not in arrays:
                    arrays[name] = diag[name].copy()
        tables = native.scheme_state['ruc_table_words']
        tbq = saturation_table(arrays['soilt'].buffer.context)
        present = {'pattern_spp_lsm': not ice, 'field_sf': not ice}
        if ice:
            solve_ice(arrays, scalars, tables, tbq, solve_cells=nc, present=present)
            blend = {'xice_p': arrays['xice'], 'sfc_albedo_p': arrays['alb'], 'sfc_emiss_p': arrays['emiss']}
            for name in ruc.ICE_FINISH_FIELDS:
                blend[name + '_p'] = arrays['soilt' if name == 'tsk' else name]
            sea = {name + '_sea': prepared[name + '_sea'] for name in ruc.ICE_FINISH_FIELDS}
            boundaries.finish_seaice(blend, sea, solve_cells=nc, threshold=threshold,
                                     fractional=options['config_frac_seaice'],
                                     mynn=options['config_sfclayer_scheme'].strip() == 'sf_mynnsfclay')
            for name in ('cqs', 'cpm', 'chs2', 'cqs2', 'qgh'):
                diag[name][...] = arrays[name]
        else:
            solve_land(arrays, scalars, tables, tbq, solve_cells=nc, present=present)
        for field in ruc.ARGUMENTS[entry]:
            name = field['name']
            if not field['dim'] or name in ('pattern_spp_lsm', 'field_sf'):
                continue
            if (ice and name not in ruc.ICE_COPYBACK) or (not ice and field['intent'] == 'in'):
                continue
            target = sfc[ruc.SFC_BINDINGS[name]] if name in ruc.SFC_BINDINGS else diag[ruc.DIAG_BINDINGS.get(name, name)]
            target[...] = arrays[name]
        if ice:
            diag['flqc'][...] = arrays['flhc']
            sfc['sfc_albbck'][...] = arrays['albbck']
        elif micro:
            diag['sr'][...] = arrays['frzfrac']
        return arrays
