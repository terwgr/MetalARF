"""Lazy native Registry and prepared-array views for Metal slot callbacks.

These views share the session's resident buffers. Constants, scalar manager
state and calendar metadata still come from the owned host framework.
"""
from collections.abc import Mapping
from copy import copy
from .device_pools_v842 import _PoolViews, _PreparedViews, _ScalarViews, _ConstantViews, _ProfileViews
from .route_table import route_row


class MetalSlotPools(Mapping):
    def __init__(self, session):
        self.session, self.native = session, session.native
        self.cp = session.pools.arrays
        self.resident = True
        self.level = 1
        self._views = {'prepared': _PreparedViews(self),
                       'scalar_state': _ScalarViews(self.native),
                       'constants': _ConstantViews(self.native)}

    def __iter__(self):
        return iter(dict.fromkeys((*self.session.layout.pools,
            *self.session.layout.dimensions, 'prepared', 'scalar_state',
            'constants', 'calendar', 'configs')))

    def __len__(self):
        return sum(1 for _ in self)

    def __getitem__(self, name):
        if name in self.session.layout.dimensions:
            return self.session.layout.dimensions[name]
        if name == 'configs':
            return self.session.layout.configs
        if name == 'calendar':
            return {key: self.native.get_table('mpas_atmphys_manager.' + key)[0][()]
                    for key in ('curr_julday', 'gmt', 'julday')}
        if name not in self._views:
            if name not in self.session.layout.pools:
                raise KeyError(name)
            self._views[name] = _PoolViews(self, name)
        return self._views[name]

    def field(self, pool, name, *, resident=False):
        if name in self.session.layout.dimensions:
            return self.session.layout.dimensions[name]
        return self.session.pools.field(pool, name,
            level=self.level if pool == 'state' else 1, full=resident)

    def prepared(self, name):
        return self.session.pools.prepared(name)

    def for_slot(self, call):
        """Apply the native descriptor's aliases over the same resident arrays."""
        descriptor = route_row(self.session.config)['slots'][call.slot]['cuda_entries'][call.scheme]
        views=copy(self)
        views.level=2 if call.slot=='microphysics' else 1
        views._views={'prepared':_PreparedViews(views),
                      'scalar_state':self._views['scalar_state'],
                      'constants':_ConstantViews(self.native,descriptor.get('constant_aliases',{}))}
        for name,aliases in descriptor.get('profile_pools',{}).items():
            views._views[name]=_ProfileViews(views,aliases)
        if descriptor.get('profiles'):
            views._views['atm_input']=_ProfileViews(views,descriptor['profiles'])
        return views
