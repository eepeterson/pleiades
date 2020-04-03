import os
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from pleiades import ArbitraryPoints, RectangularCoil, MagnetRing, Device

class TREXCoil(ArbitraryPointsSet):
    def __init__(self, z0, **kwargs):
        self.z0 = z0
        self._dr = 0.0134166
        self._dz = 0.0134166
        self._npts = 88
        r = 2.00467 + np.linspace(0, .067083, 6)
        z = self.z0 + np.linspace(-.105469,.105469, 16)
        rz_pts = [(ri, zi) for ri in r[:5] for zi in z]
        rz_pts.extend([(r[5], zi) for zi in (z[::2] + z[1::2])/2])
        super().__init__(np.array(rz_pts), **kwargs)

    @property
    def z0(self):
        return self._z0

    @property
    def dr(self):
        return self._dr

    @property
    def dz(self):
        return self._dz

    @property
    def npts(self):
        return self._npts

    @property
    def rz_pts(self):
        pass

    @property
    def verts(self):
        pass

    @property
    def area(self):
        return self.dr*self.dz*self.npts

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

    @rz_pts.setter
    def rz_pts(self, rz_pts):
        msg = ('TREX coils cannot have their points changed except through the '
               'z0 attribute')
        raise NotImplementedError(msg)


class Dipole(Component):
    """Internal dipole Magnet comprised of 2 cylindrical SmCo magnets.

    Attributes:
        magnets (list): list of Magnet objects comprising this instance
        patches (list of matplotlib.patches.Polygon instances): patches representing the vessel magnets
    """

    def __init__(self, **kwargs):
        super(Dipole,self).__init__()
        r0,z0 = kwargs.pop("loc",(0,0))
        muhat = kwargs.pop("muhat",0)
        labels = kwargs.pop("labels",["dipole"])
        nprocs = kwargs.pop("nprocs",[1])
        currents = kwargs.pop("currents",[2901.0])
        patch_mask = kwargs.pop("patch_mask",[0])
        # Build internal dipole magnets
        width = (2.75 / 2 - .125) * .0254
        height = 2.5 / 2 * .0254
        delta = (1.25 / 2 + .125) * .0254
        r1 = r0 + .125 * .0254 + width / 2.0  # + delta*np.sin(np.pi*mu_hat/180.0)
        r2 = r1  # rho0 - delta*np.sin(np.pi*mu_hat/180.0)
        z1 = z0 + delta  # *np.cos(np.pi*mu_hat/180.0)
        z2 = z0 - delta  # *np.cos(np.pi*mu_hat/180.0)
        m1 = MagnetGroup(rz_pts=[(r1,z1),(r2,z2)],mu_hats=[muhat,muhat],height=height,width=width,current=currents[0],**kwargs)


class BRB(Device):
    def __init__(self):
        # Global default patch settings
        kwargs = {'fc': '.35', 'ec': 'k'}

        # Set TREX HH coil default parameters
        z0 = 1.1
        self.hh_n = TREXCoil(z0, **kwargs)
        self.hh_s = TREXCoil(-z0, **kwargs)

        # Set LTRX mirror coil default parameters
        r0, z0 = 0.185725, 1.6367
        dr, dz = 0.010583333, 0.01031667
        ltrx_n = RectangularCoil(r0, z0, nr=10, nz=13, dr=dr, dz=dz, **kwargs)
        ltrx_s = RectangularCoil(r0, -z0, nr=10, nz=13, dr=dr, dz=dz, **kwargs)
        self.ltrx_n = ltrx_n
        self.ltrx_s = ltrx_s

        # Set default magnet cage parameters
        kwargs.update({'current': 2710.68, 'height': 0.0254, 'width': 0.0381})

        # Build array of r0, z0, mu_hat values for all 36 magnet rings
        centroid_radius = 1.514475
        theta = np.linspace(7.5, 172.5, 34)
        rzmu = np.empty((36, 3))
        rzmu[0, :] = [0.0768, 1.5117, 0.]
        rzmu[1:-1, 0] = centroid_radius*np.sin(math.pi*theta/180)
        rzmu[1:-1, 1] = centroid_radius*np.cos(math.pi*theta/180)
        rzmu[1:-1, 2] = theta + np.mod(np.arange(1, 35), 2)*180
        rzmu[-1, :] = [0.0768, -1.5117, 0.]

        for i, (r0, z0, mu_hat) in enumerate(rzmu):
            if np.mod(i, 2):
                kwargs['fc'] = 'r'
            else:
                kwargs['fc'] = 'b'
            setattr(self, f'mr{i+1}', MagnetRing(r0, z0, **kwargs)) 

    def add_cathode(self):
        raise NotImplementedError("Can't add cathodes to BRB yet")

    def add_anode(self):
        raise NotImplementedError("Can't add anode to BRB yet")

    def add_sweep(self,center,r,theta1,theta2,width=None,**kwargs):
        self.patches.append(Wedge(center,r,theta1,theta2,width=width,**kwargs))

