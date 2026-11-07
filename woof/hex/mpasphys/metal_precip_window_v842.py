"""Native RP6 surface precipitation window over resident Metal Registry views."""
import numpy as np
from .precip_window_v842 import SurfacePrecipitationWindow,WINDOW_ARRAYS,SNAPSHOT_PREFIX,SNOW_GRAUPEL_SCHEMES
from .metal_tendencies_v842 import _launch


class MetalSurfacePrecipitationWindow(SurfacePrecipitationWindow):
    """Reuse native field selection and checkpoint validation with Metal kernels."""
    def __init__(self,session):
        self.session=session;self.native=session.native
        self.cells=session.layout.dimensions['nCellsSolve'];self.cp=session.pools.arrays
        if self.cells<1:raise ValueError('the precipitation window needs solve cells')
        selectors=session.config.selectors
        self.have_nc=selectors['microphysics']!='off'
        self.have_c=selectors['convection']!='off'
        self.have_sg=selectors['microphysics'] in SNOW_GRAUPEL_SCHEMES
        self.window={name:self.cp.zeros((self.cells,),np.float32) for name in WINDOW_ARRAYS}
        self.steps=self.cp.zeros((self.cells,),np.int32)
        self.saved=None

    def _solve_words(self,name):
        return self.session.pools.field('diag_physics',name)

    def open(self,call=None):
        if self.saved is not None:raise RuntimeError('surface precipitation window is already open')
        self.saved={name:self._solve_words(name).copy() for name in self.present()}
        for name in self.saved:self._solve_words(name)[...]=self.window[name]

    def accumulate(self,call=None):
        sr=self._solve_words('sr');present=self.present()
        fields={name:self._solve_words(name) if name in present else sr
                for name in ('rainncv','raincv','snowncv','graupelncv')}
        resources=dict(fields,sr=sr,nsteps_bl=self.steps,
                       **{name+'_bl':value for name,value in self.window.items()})
        _launch('hexv842_mpas_precip_window_add_f32',self.cells,self.cp.context,resources,
                dict(n=np.int32(self.cells),have_nc=np.int32(self.have_nc),
                     have_c=np.int32(self.have_c),have_sg=np.int32(self.have_sg)))

    def close(self,call=None):
        if self.saved is None:raise RuntimeError('surface precipitation window is not open')
        for name,words in self.saved.items():self._solve_words(name)[...]=words
        self.saved=None
        resources={name+'_bl':self.window[name] for name in WINDOW_ARRAYS if name!='sr'}
        _launch('hexv842_mpas_precip_window_reset_f32',self.cells,self.cp.context,
                dict(resources,nsteps_bl=self.steps),dict(n=np.int32(self.cells)))

    def restart_state(self):
        if self.saved is not None:raise RuntimeError('surface precipitation window is open')
        return {**{SNAPSHOT_PREFIX+name:value.to_numpy() for name,value in self.window.items()},
                SNAPSHOT_PREFIX+'nsteps':self.steps.to_numpy()}

    snapshot=restart_state
    restore_restart_state=SurfacePrecipitationWindow.restore
