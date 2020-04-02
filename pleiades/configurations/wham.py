import numpy as np
import os
import warnings
from matplotlib.patches import Polygon, Wedge
from matplotlib.collections import PatchCollection
import matplotlib as mpl

from pleiades import (Current, CurrentGroup, Magnet, Component, ZSymmCoilSet,
                      MagnetGroup, CurrentArray, Configuration)

class PhilipsMRI(object):
    def __init__(self, loc, current):
        rho0, z0 = loc
        delta_rho = .01
        delta_z = .01
        nz = 1.5 // delta_z
        nrho = .05 // delta_rho
        z1 = nz / 2 * delta_z
        ## dimensions of cryostats
        inner_r = .930 / 2.0
        outer_r = 1.88 / 2.0
        length = 1.618
        verts = np.array([[inner_r,z0+length/2.0],[outer_r,z0+length/2.0],[outer_r,z0-length/2.0],[inner_r,z0-length/2.0]])
        self.cryo = Polygon(verts,closed=True,fc="None",ec="k",lw=2,joinstyle="round")
        self.coil = CurrentArray((rho0,z0-z1),nrho,nz,delta_rho,delta_z,current,fc=".45",units="m")
        self.patches = [self.cryo,self.coil.patch]

    def get_current_tuples(self,frame="rhoz",units="m"):
        return self.coil.get_current_tuples(frame=frame,units=units)


def build_wham():
    rho0 = 40
    n_rho = 20
    n_z = 100
    delta_rho = 1
    delta_z = 1
    coil_1 = CurrentArray((rho0, -325), n_rho, n_z, delta_rho, delta_z, 1500, units="cm")
    coil_2 = CurrentArray((rho0, -215), n_rho, n_z, delta_rho, delta_z, 500, units="cm")
    coil_3 = CurrentArray((rho0, -105), n_rho, n_z, delta_rho, delta_z, 500, units="cm")
    coil_4 = CurrentArray((rho0, 5), n_rho, n_z, delta_rho, delta_z, 500, units="cm")
    coil_5 = CurrentArray((rho0, 115), n_rho, n_z, delta_rho, delta_z, 500, units="cm")
    coil_6 = CurrentArray((rho0, 225), n_rho, n_z, delta_rho, delta_z, 1500, units="cm")
    coil_7 = CurrentArray((rho0 / 2.0, -395), 10, 50, delta_rho, delta_z, 30000, units="cm")
    coil_8 = CurrentArray((rho0 / 2.0, 345), 10, 50, delta_rho, delta_z, 30000, units="cm")
    current_objs = [coil_1, coil_2, coil_3, coil_4, coil_5, coil_6, coil_7, coil_8]
    patches = [coil_1.patch, coil_2.patch, coil_3.patch, coil_4.patch, coil_5.patch, coil_6.patch, coil_7.patch,
               coil_8.patch]

    return current_objs, patches
