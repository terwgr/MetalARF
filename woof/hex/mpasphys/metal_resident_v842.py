"""Resident MPAS v8.4.2 native pool preparation for Metal physics slots."""
from __future__ import annotations
import numpy as np
from ..metal_dynamics_v841 import _context, _i
from .metal_preparation_v842 import prepare_mass, prepare_interfaces, prepare_numbers
from .metal_tendencies_v842 import _launch


class MetalResidentPreparation:
    """Bind the source MPAS_to_physics transaction to initialized native pools."""
    def __init__(self, session, config):
        self.session,self.config=session,config
        self.prepared=session.pools.prepared
        self.names=tuple(next(row['components'] for row in session.layout.fields
                              if row['pool']=='state' and row['name']=='scalars'))
        if config.moist_physics and (not self.names or self.names[0]!='qv'):
            raise ValueError('resident MPAS preparation requires qv first')
        self.active=False
        self.rho_dry=None
        self.extra_species={}
        self.radshared={}

    def allocation_stems(self):
        stems=['forall_physics','cloudiness','radiation_sw','radiation_lw','sfclayer',
               'lsm','pbl','gwdo','convection','microphysics']
        if self.config.active('seaice'): stems.append('seaice')
        return stems

    def activate(self):
        """Allocate native module globals after framework physics initialization."""
        if self.active:
            return
        native=self.session.native
        if not native._initialized:
            raise RuntimeError('MPAS framework physics must initialize before resident preparation')
        for stem in self.allocation_stems():
            native.call('allocate_'+stem)
        self.active=True

    def prepare(self, call):
        """MPAS_to_physics: prepare named native variables at the driver boundary."""
        if not self.active:
            self.activate()
        p=self.session.pools; native=self.session.native
        nz=self.session.layout.dimensions['nVertLevels']; nc=self.session.layout.dimensions['nCellsSolve']
        field=p.field; prepared=self.prepared
        constants={name:np.float32(native.get_constant(name)) for name in ('r_v','r_d','gravity','degrad')}
        inputs={'rho_zz':field('state','rho_zz'),'theta_m':field('state','theta_m'),
                'scalars':field('state','scalars'),'exner':field('diag','exner'),
                'pressure_base':field('diag','pressure_base'),'pressure_p':field('diag','pressure_p'),
                'zgrid':field('mesh','zgrid'),'zz':field('mesh','zz')}
        mass=prepare_mass(fields=inputs,scalar_names=self.names,rvord=np.float32(constants['r_v']/constants['r_d']))
        self.rho_dry=mass['rho_dry']
        for name in ('zz_p','rho_p','th_p','t_p','pi_p','pres_p','zmid_p','dz_p'):
            prepared(name)[:nz][...] = mass[name]
        species={}
        known=('qv','qc','qr','qi','qs','qg','nc','ni','nifa','nwfa')
        for index,name in enumerate(self.names):
            value=mass['qout'][index]
            if name in known:
                try:
                    target=prepared(name+'_p')[:nz]
                except RuntimeError:
                    target=value
                else:
                    target[...] = value
                species[name]=target
            else:
                species[name]=value
        self.extra_species=species
        for target,source in (('u_p','uReconstructZonal'),('v_p','uReconstructMeridional')):
            prepared(target)[:nz][...] = field('diag',source)
        names=('w_p','z_p','t2_p','pres2_p','pres_hyd_p','pres2_hyd_p','pres_hydd_p',
               'pres2_hydd_p','psfc_p','psfc_hyd_p','psfc_hydd_p','znu_p','znu_hyd_p')
        interface={name:prepared(name) for name in names}
        interface.update(rho_zz=inputs['rho_zz'],zz=inputs['zz'],qv_p=species['qv'],
            pres_p=mass['pres_p'],t_p=mass['t_p'],rho_p=mass['rho_p'],dz_p=mass['dz_p'],
            zgrid=inputs['zgrid'],w=field('state','w'),plrad=field('diag_physics','plrad'),
            invalid=mass['invalid'],pressure_p=inputs['pressure_p'],pressure_base=inputs['pressure_base'],
            weights_minus=prepared('fzm_p'),weights_plus=prepared('fzp_p'))
        prepare_interfaces(fields=interface,nlev=nz,ncells=nc,gravity=constants['gravity'])
        field('diag','surface_pressure')[...] = prepared('psfc_p')
        flags={name:bool(native.get_variable('f_'+name)[0]) for name in ('nc','ni','nifa','nwfa')}
        number_schemes = getattr(self.session.driver_api, 'number_carrier_schemes', None)
        if number_schemes is None:
            micro, pbl = call.options['config_microp_scheme'], call.options['config_pbl_scheme']
        else:
            micro, pbl = number_schemes(call.options)
        numbers=prepare_numbers(species=species,microphysics=micro,
            pbl=pbl,scalar_flags=flags,n_cells_solve=nc)
        for name,value in numbers.items(): prepared(name+'_p')[:nz][...] = value
        _launch('hexv842_mpas_latitude_f32',nc,_context(inputs['rho_zz']),
                dict(lat=field('mesh','latCell'),out=prepared('xlat_p')),
                dict(ncells=_i(nc),degrad=constants['degrad']))
        return mass['invalid']


    def cloudiness(self, call):
        """Run native cloud fraction against prepared radiation carriers."""
        from dataclasses import replace
        if not self.active:
            raise RuntimeError('cloudiness requires a preceding MPAS_to_physics call')
        p=self.session.pools; nz=self.session.layout.dimensions['nVertLevels']
        prepared={name:self.prepared(name+'_p')[:nz] for name in ('qv','qc','qi','qs')}
        radiation={name:self.prepared(name+'_p')[:nz] for name in ('qvrad','qcrad','qirad','qsrad','cldfrac')}
        radiation.update({name:self.prepared(name+'_p') for name in ('dx','xland')})
        from .rrfs.metal_cloudiness import CLOUD_SCHEMES
        if call.scheme in CLOUD_SCHEMES:
            from .rrfs.metal_cloudiness import cloudiness_slot
            diagnostics = ['cldfrac']
            if call.scheme == 'cld_fraction_mynn':
                diagnostics += ['cldfrac_bl', 'qc_bl', 'qi_bl']
                radiation.update({name: self.prepared(name + '_p')[:nz] for name in ('qcbl', 'qibl')})
            if call.options['config_convection_scheme'].strip() == 'cu_grell_freitas_li':
                diagnostics += ['cldfrac_cu', 'qc_cu', 'qi_cu']
                radiation.update({name: self.prepared(name + '_p')[:nz] for name in ('qccu', 'qicu')})
            pools = {'prepared': prepared, 'radiation_carriers': radiation,
                     'mesh': {'meshDensity': p.field('mesh', 'meshDensity')},
                     'sfc_input': {'xland': p.field('sfc_input', 'xland')},
                     'diag_physics': {name: p.field('diag_physics', name)[:nz] for name in diagnostics},
                     'len_disp': self.session.layout.configs['config_len_disp'],
                     'radshared': self.radshared, 'itimestep': call.itimestep}
            result = cloudiness_slot(replace(call, pools=pools))
            self.radshared.update(radiation)
            return result
        from .metal_cloud_v842 import cloudiness_slot
        prepared.update({name:self.prepared(name+'_p')[:nz] for name in ('t','pres')})
        pools={'prepared':prepared,'radiation_prepared':radiation,
               'mesh':{'meshDensity':p.field('mesh','meshDensity')},
               'sfc_input':{'xland':p.field('sfc_input','xland')},
               'diag_physics':{'cldfrac':p.field('diag_physics','cldfrac')},
               'config_len_disp':self.session.layout.configs['config_len_disp'],
               'radshared':self.radshared}
        from .radshared_v842 import icloud_bl_enabled
        if icloud_bl_enabled(call.options):
            pools['diag_physics'].update({name:p.field('diag_physics', name)[:nz]
                for name in ('qc_bl','qi_bl','cldfrac_bl')})
        result=cloudiness_slot(replace(call,pools=pools))
        self.radshared.update(radiation)
        return result

    def surface_layer(self, call):
        """Execute sf_mynn with native prepared profiles and same-step sea carriers."""
        from dataclasses import replace
        from .metal_sf_mynn_v842 import MetalMpasSfMynn, CONSTANTS, _SEA_HANDOFF
        from ..physics_suites import load_table
        p=self.session.pools; native=self.session.native
        if not self.active:
            raise RuntimeError('surface layer requires a preceding MPAS_to_physics call')
        class Pool:
            def __init__(self, name): self.name=name
            def __getitem__(self, name): return p.field(self.name,name)
        descriptor=load_table()['mpasphys_v842']['slots']['surface_layer']['cuda_entries']['sf_mynn']
        aliases=descriptor['constant_aliases']
        constants={name:native.get_constant(aliases.get(name,name)) for name in CONSTANTS}
        profiles={name:self.prepared(source)[:self.session.layout.dimensions['nVertLevels']]
                  for name,source in descriptor['profiles'].items()}
        prepared={'psfc_p':self.prepared('psfc_p')}
        if call.options['config_frac_seaice']:
            prepared.update({name+'_sea':self.prepared(name+'_sea') for name in _SEA_HANDOFF})
        threshold=native.get_variable('xice_threshold')
        if threshold is None: raise RuntimeError('MPAS surface ice threshold is not initialized')
        pools={'mesh':Pool('mesh'),'diag_physics':Pool('diag_physics'),'sfc_input':Pool('sfc_input'),
               'atm_input':profiles,'constants':constants,'configs':self.session.layout.configs,
               'scalar_state':{'xice_threshold':threshold[0][()]},'prepared':prepared}
        if not hasattr(self,'_sf_mynn'):
            self._sf_mynn=MetalMpasSfMynn(_context(prepared['psfc_p']))
        return self._sf_mynn.launch_slot(replace(call,pools=pools))

    def gwdo(self, call):
        """Execute native YSU GWDO against prepared and Registry pool views."""
        from .metal_gwdo_v842 import run_gwdo
        if call.scheme != 'bl_ysu_gwdo':
            raise ValueError(f'MPAS Metal GWDO scheme {call.scheme!r} is not implemented')
        if not self.active:
            raise RuntimeError('GWDO requires a preceding MPAS_to_physics call')
        p=self.session.pools; native=self.session.native
        class Pool:
            def __init__(self,name): self.name=name
            def __getitem__(self,name): return p.field(self.name,name)
        prepared={name:self.prepared(name) for name in
                  ('u_p','v_p','t_p','qv_p','pres_hydd_p','pres2_hydd_p','pi_p','zmid_p')}
        dt=native.get_variable('dt_pbl')
        if dt is None: raise RuntimeError('MPAS PBL interval is not initialized')
        constants={key:native.get_constant(source) for key,source in
                   (('gravity','gravity'),('cp','cp'),('rd','r_d'),('fv','ep_1'),('pi','pii'))}
        return run_gwdo(prepared=prepared,mesh=Pool('mesh'),sfc_input=Pool('sfc_input'),
                        tend_physics=Pool('tend_physics'),diag_physics=Pool('diag_physics'),
                        dt_pbl=float(dt[0][()]),config_len_disp=self.session.layout.configs['config_len_disp'],
                        constants=constants)
