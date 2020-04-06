import os
import warnings
import math
import numpy as np
import matplotlib as mpl
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.collections import PatchCollection

from pleiades import ArbitraryPoints, RectangularCoil, MagnetRing, Device

class TREXCoil(ArbitraryPoints):
    """TREX STUFF"""
    _RMIN = 2.00467
    _DR = 0.0134166
    _DZ = 0.0140625
    _COND_AREA = _DR*_DR
    _NPTS = 88
    _Z_REL_POS = np.linspace(-.105469,.105469, 16)
    _R_REL_POS = np.linspace(0, .067083, 6)

    _codes = [Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CLOSEPOLY]

    def __init__(self, z0=1.1, **kwargs):
        r = self._RMIN + self._R_REL_POS
        z = z0 + self._Z_REL_POS
        rz_pts = [(ri, zi) for ri in r[:5] for zi in z]
        rz_pts.extend([(r[5], zi) for zi in (z[::2] + z[1::2])/2])
        super().__init__(np.array(rz_pts), **kwargs)

    @property
    def z0(self):
        return self._z0

    @property
    def dr(self):
        return self._DR

    @property
    def dz(self):
        return self._DZ

    @property
    def npts(self):
        return self._NPTS

    @property
    def _verts(self):
        rpts, zpts = self._rz_pts[:, 0], self._rz_pts[:, 1]
        rmin, rmax = np.amin(rpts) - self._DR/2, np.amax(rpts) + self._DR/2
        zmin, zmax = np.amin(zpts) - self._DZ/2, np.amax(zpts) + self._DZ/2
        verts = np.array([[rmin, zmin],
                          [rmin, zmax],
                          [rmax, zmax],
                          [rmax, zmin],
                          [rmin, zmin]])
        return verts

    @property
    def patch(self):
        return patches.PathPatch(Path(self._verts, self._codes))

    @property
    def area(self):
        return self._COND_AREA*self._NPTS

    @property
    def total_current(self):
        return self.current*np.sum(self.weights)

    @property
    def current_density(self):
        return self.total_current / self.area

    @z0.setter
    def z0(self, z0):
        delta_z = np.array([0, z0 - self.z0]).reshape((1, 2))
        self._rz_pts = self._rz_pts + delta_z
        self._z0 = z0

    @ArbitraryPoints.rz_pts.setter
    def rz_pts(self, rz_pts):
        msg = ('rz_pts for TREXCoil must be changed using the z0 attribute')
        raise NotImplementedError(msg)


#class Dipole(Component):
#    """Internal dipole Magnet comprised of 2 cylindrical SmCo magnets.
#
#    Attributes:
#        magnets (list): list of Magnet objects comprising this instance
#        patches (list of matplotlib.patches.Polygon instances): patches representing the vessel magnets
#    """
#
#    def __init__(self, **kwargs):
#        super(Dipole,self).__init__()
#        r0,z0 = kwargs.pop("loc",(0,0))
#        muhat = kwargs.pop("muhat",0)
#        labels = kwargs.pop("labels",["dipole"])
#        nprocs = kwargs.pop("nprocs",[1])
#        currents = kwargs.pop("currents",[2901.0])
#        patch_mask = kwargs.pop("patch_mask",[0])
#        # Build internal dipole magnets
#        width = (2.75 / 2 - .125) * .0254
#        height = 2.5 / 2 * .0254
#        delta = (1.25 / 2 + .125) * .0254
#        r1 = r0 + .125 * .0254 + width / 2.0  # + delta*np.sin(np.pi*mu_hat/180.0)
#        r2 = r1  # rho0 - delta*np.sin(np.pi*mu_hat/180.0)
#        z1 = z0 + delta  # *np.cos(np.pi*mu_hat/180.0)
#        z2 = z0 - delta  # *np.cos(np.pi*mu_hat/180.0)
#        m1 = MagnetGroup(rz_pts=[(r1,z1),(r2,z2)],mu_hats=[muhat,muhat],height=height,width=width,current=currents[0],**kwargs)


class BRB(Device):
    def __init__(self):
        # Global default patch settings
        super().__init__()

        # Set TREX HH coil default parameters
        z0 = 1.1
        patch_kw = {'fc': '.35', 'ec': 'k'}
        self.hh_n = TREXCoil(z0, patch_kw=patch_kw)
        self.hh_s = TREXCoil(-z0, patch_kw=patch_kw)

        # Set LTRX mirror coil default parameters
        r0, z0 = 0.185725, 1.6367
        dr, dz = 0.010583333, 0.01031667
        ltrx_n = RectangularCoil(r0, z0, nr=10, nz=13, dr=dr, dz=dz,
                                 patch_kw=patch_kw)
        ltrx_s = RectangularCoil(r0, -z0, nr=10, nz=13, dr=dr, dz=dz,
                                 patch_kw=patch_kw)
        self.ltrx_n = ltrx_n
        self.ltrx_s = ltrx_s

        # Build array of r0, z0, mu_hat values for all 36 magnet rings
        centroid_radius = 1.514475
        theta = np.linspace(7.5, 172.5, 34)
        rzmu = np.empty((36, 3))
        rzmu[0, :] = [0.0768, 1.5117, 0.]
        rzmu[1:-1, 0] = centroid_radius*np.sin(math.pi*theta/180)
        rzmu[1:-1, 1] = centroid_radius*np.cos(math.pi*theta/180)
        rzmu[1:-1, 2] = theta + np.mod(np.arange(1, 35), 2)*180
        rzmu[-1, :] = [0.0768, -1.5117, 0.]

        # Set default magnet cage parameters
        mag_kw = {'current': 2710.68, 'height': 0.0254, 'width': 0.0381}

        for i, (r0, z0, mu_hat) in enumerate(rzmu):
            kwargs = {'r0': r0, 'z0': z0, 'mu_hat': mu_hat}
            kwargs.update(mag_kw)
            if np.mod(i, 2):
                patch_kw = {'fc': 'r'}
            else:
                patch_kw = {'fc': 'b'}
            kwargs['patch_kw'] = patch_kw
            setattr(self, f'mr{i+1}', MagnetRing(**kwargs)) 

    def add_cathode(self):
        raise NotImplementedError("Can't add cathodes to BRB yet")

    def add_anode(self):
        raise NotImplementedError("Can't add anode to BRB yet")

    def add_sweep(self, center,r,theta1,theta2,width=None,**kwargs):
        self.patches.append(patches.Wedge(center,r,theta1,theta2,width=width,**kwargs))
