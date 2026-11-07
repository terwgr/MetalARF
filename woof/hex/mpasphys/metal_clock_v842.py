"""Checkpoint the owned MPAS framework clock and Registry text levels."""
from fractions import Fraction


def _time_identity(value):
    return {'year':value.year,'month':value.month,'day':value.day,
            'seconds':str(value.seconds),'calendar':value.calendar}


class MetalClockCheckpoint:
    def __init__(self,native):
        self.native=native

    def restart_state(self):
        clock=self.native.clock_state
        return {'start':_time_identity(clock.start_time), 'stop':_time_identity(clock.stop_time), 'dt':str(clock.dt),
                'elapsed':str(clock.elapsed),
                'text':{f'{pool}.{name}.{level}':self.native.get_text_field(pool,name,level=level)
                        for (pool,name),levels in self.native._text_fields.items()
                        for level in range(1,levels+1)}}

    def restore_restart_state(self,state):
        clock=self.native.clock_state
        if state['start']!=_time_identity(clock.start_time) or state['stop']!=_time_identity(clock.stop_time) or state['dt']!=str(clock.dt):
            raise ValueError('MPAS framework clock configuration differs')
        elapsed=Fraction(state['elapsed'])
        expected={f'{pool}.{name}.{level}' for (pool,name),levels in self.native._text_fields.items()
                  for level in range(1,levels+1)}
        if elapsed<0 or set(state['text'])!=expected:
            raise ValueError('invalid MPAS framework clock checkpoint')
        for key,value in state['text'].items():
            if not isinstance(value,str) or len(value.encode('ascii'))>512 or '\0' in value:
                raise ValueError(f'invalid MPAS text field {key}')
        for key,value in state['text'].items():
            pool,name,level=key.rsplit('.',2)
            self.native.put_text_field(pool,name,value,level=int(level),nlevels=self.native._text_fields[pool,name])
        clock.elapsed=elapsed
        clock.current_time=clock.start_time.add(elapsed)
