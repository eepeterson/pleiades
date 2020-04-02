import os
import warnings
import numpy as np
import matplotlib as mpl
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

from pleiades import ArbitraryPointsSet, RectangularCoil, Device

class TREXCoil(ArbitraryPointsSet):
    def __init__(self, z0, **kwargs):
        self.z0 = z0
        Rarr = 2.00467 + np.linspace(0, .067083, 6)
        Zarr = np.linspace(-.105469,.105469, 16)
        rz_pts = []
        for i, r in enumerate(Rarr):
            if i == 5:
                rz_pts.extend([(r, z0 + z) for z in (Zarr[0::2]+Zarr[1::2])/2])
            else:
                rz_pts.extend([(r, z0 + z) for z in Zarr])
        rz_pts = np.array(rz_pts)

    @property
    def z0(self):
        return self._z0

    @property
    def npts(self):
        pass

    @property
    def rz_pts(self):
        pass

    @property
    def verts(self):
        pass

    @property
    def area(self):
        pass

    @property
    def total_current(self):
        pass

    @property
    def current_density(self):
        pass

    @z0.setter
    def z0(self, z0):
        self._z0 = z0

    @rz_pts.setter
    def (self, rz_pts):
        msg = ('TREX coils cannot have their points changed except through the
               z0 attribute')
        raise NotImplementedError(msg)


class VesselMagnets(Component):
    def __init__(self,**kwargs):
        super(VesselMagnets,self).__init__()
        labels = kwargs.pop("labels",["Npole","bulk","Spole"])
        nprocs = kwargs.pop("nprocs",[1,12,1])
        currents = kwargs.pop("currents",[2710.68,2710.68,2710.68])
        patch_mask = kwargs.pop("patch_mask",[0,0,0])
        height = 1 * .0254
        width = 1.5 * .0254
        # first group
        z = 1.5117
        r = .0768
        kwargs.update({"fc":"b"})
        m1 = MagnetGroup(rz_pts=[(r,z)],mu_hats=[0],height=height,width=width,**kwargs)
        # second group
        R = 1.514475
        Theta = np.linspace(7.5, 172.5, 34)
        rpts,zpts = R*np.sin(np.deg2rad(Theta)),R*np.cos(np.deg2rad(Theta))
        rz_pts = np.vstack((rpts,zpts)).T
        mu_hats = Theta + np.mod(np.arange(1, 35), 2) * 180
        m2 = MagnetGroup(rz_pts=rz_pts,mu_hats=mu_hats,height=height,width=width,**kwargs)
        for m_obj in m2.obj_list[::2]:
            m_obj.patchkwargs["fc"]="r"
        # third group
        z = -1.5117
        r = .0768
        kwargs.update({"fc":"r"})
        m3 = MagnetGroup(rz_pts=[(r,z)],mu_hats=[0],height=height,width=width,**kwargs)
        self.groups = [m1,m2,m3]
        self.labels = labels
        self.nprocs = nprocs
        self.patch_mask = patch_mask
        self.currents = currents
        self.update_patches()

    @Component.patches.getter
    def patches(self):
        plist = [group.patches for group,mask in zip(self._groups,self._patch_mask) if not mask]
        return [p for sublist in plist for p in sublist]

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
        self.groups = [m1]
        self.labels = labels
        self.nprocs = nprocs
        self.patch_mask = patch_mask
        self.currents = currents
        self.update_patches()
        
    @Component.patches.getter
    def patches(self):
        plist = [group.patches for group,mask in zip(self._groups,self._patch_mask) if not mask]
        return [p for sublist in plist for p in sublist]

class BRB(Device):
    def __init__(self):
        # Global default patch settings
        patch_kw = {'fc': '.35', 'ec': 'k'}

        # Set TREX HH coil default parameters
        z0 = 1.1
        self.hh_n = TREXCoil(z0, **patch_kw)
        self.hh_s = TREXCoil(-z0, **patch_kw)

        # Set LTRX default parameters
        r0, z0 = 0.185725, 1.6367
        dr, dz = 0.010583333, 0.01031667
        self.ltrx_n = RectangularCoil(r0, z0, nr=10, nz=13, dr=dr, dz=dz,
                                      **patch_kw)
        self.ltrx_n = RectangularCoil(r0, -z0, nr=10, nz=13, dr=dr, dz=dz,
                                      **patch_kw)

        # Set default magnet cage parameters



    def add_cathode(self):
        raise NotImplementedError("Can't add cathodes to BRB yet")

    def add_anode(self):
        raise NotImplementedError("Can't add anode to BRB yet")

    def add_sweep(self,center,r,theta1,theta2,width=None,**kwargs):
        self.patches.append(Wedge(center,r,theta1,theta2,width=width,**kwargs))

