from infretis.core.common import compare_objects
import numpy as np
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class System:
    def __init__(self):
        self.order = None
        self.pos = None
        self.vel = None
        self.ekin = None
        self.vpot = None
        self.other = None

    def __eq__(self, other):
        attrs =  self.__dict__.keys()
        return compare_objects(self, other, attrs)


def system_from_snapshot(system, snapshot):                 
    """Create a system from a given snapshot."""            
    system_copy = system.copy()                             
    system_copy.particles.ekin = snapshot.get('ekin', None) 
    system_copy.particles.vpot = snapshot.get('vpot', None) 
    system_copy.order = snapshot.get('order', None)         
    system_copy.particles.set_pos(snapshot.get('pos', None))
    system_copy.particles.set_vel(snapshot.get('vel', None))
    return system_copy                                      

