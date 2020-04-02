import numpy as np
import os
import warnings
from matplotlib.patches import Polygon, Wedge
from matplotlib.collections import PatchCollection
import matplotlib as mpl

from pleiades import (Current, CurrentGroup, Magnet, Component, ZSymmCoilSet,
                      MagnetGroup, CurrentArray, Configuration)

class TREXcoil(CurrentGroup):
    def __init__(self,**kwargs):
        z0 = float(kwargs.pop("z0",0))
        self._z0 = z0
        Rarr = 2.00467 + np.linspace(0, .067083, 6)
        Zarr = np.linspace(-.105469,.105469, 16)
        rz_pts = []
        for i, r in enumerate(Rarr):
            if i == 5:
                rz_pts.extend([(r, z0 + z) for z in (Zarr[0::2]+Zarr[1::2])/2])
            else:
                rz_pts.extend([(r, z0 + z) for z in Zarr])
#        for i, z in enumerate(Zarr):
#            if np.mod(i, 2) == 0:
#                rz_pts.extend([(r, z0 + z) for r in Rarr])
#            else:
#                rz_pts.extend([(r, z0 + z) for r in Rarr[0:5]])
        rz_pts = np.array(rz_pts)
        super_kwargs = {"rz_pts":rz_pts,"patchcls":Polygon,"fc":".35","ec":"k"}
        super_kwargs.update(kwargs)
        super(TREXcoil,self).__init__(**super_kwargs)

    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self,new_z0):
        dz = new_z0 - self._z0
        super(TREXcoil,self).translate((0,dz))
        self._z0 = new_z0

    def build_patchargs(self,**kwargs):
        z0 = self._z0
        left,right = 2.00467, 2.00467 + .067083
        bottom, top = z0 - .105469, z0 + .105469
        return (np.array([[left,bottom],[left,top],[right,top],[right,bottom]]),)

class TREXCoils(Component):
    def __init__(self,**kwargs):
        ###### Build TREX coils
        super(TREXCoils,self).__init__()
        z0 = float(kwargs.pop("z0",1.1757))
        labels = kwargs.pop("labels",["Ncoil","Scoil"])
        currents = np.array(kwargs.pop("currents",(1,1)),dtype="float")
        nprocs = kwargs.pop("nprocs",[4,4])
        patch_mask = kwargs.pop("patch_mask",[0,0])
        grid = kwargs.pop("grid",None)
        Scoil = TREXcoil(z0=-z0,**kwargs)
        Ncoil = TREXcoil(z0=z0,**kwargs)
        self.groups = [Ncoil,Scoil]
        self.labels = labels
        self.currents = currents
        self.nprocs = nprocs
        self.patch_mask = [0,0]

class LTRXCoils(ZSymmCoilSet):
    def __init__(self,**kwargs):
        dr,dz = 0.010583333,0.01031667
        nr,nz = 10,13
        r0,z0 = 0.185725,1.6367
        super_kwargs = {"r0":r0,"z0":z0,"dr":dr,"dz":dz,"labels":["Scoil","Ncoil"],
                "patchcls":Polygon,"fc":".35","ec":"k"}
        super_kwargs.update(kwargs)
        super(LTRXCoils,self).__init__(**super_kwargs)

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

class BRB(Configuration):
    def __init__(self,**kwargs):
        super(BRB,self).__init__()
        self.add_component(TREXCoils(),"trex")
        self.add_component(LTRXCoils(),"ltrx")
        self.add_component(VesselMagnets(),"vessel_mags")
        self.grid = kwargs.pop("grid",None)
        self.artists = [Wedge((0,0),1.556,0,360,width=.032,fc=".35",ec="k",zorder=100)]

    def add_cathode(self):
        raise NotImplementedError("Can't add cathodes to BRB yet")

    def add_anode(self):
        raise NotImplementedError("Can't add anode to BRB yet")

    def add_sweep(self,center,r,theta1,theta2,width=None,**kwargs):
        self.patches.append(Wedge(center,r,theta1,theta2,width=width,**kwargs))

class LTRX(Configuration):
    def __init__(self,**kwargs):
        super(LTRX,self).__init__()
        zc = 2.811
        self.add_component(CoilPack(r0=.286,z0=1.173-zc,nr=16,nz=16,dr=0.0135,dz=0.0135,fc=".35",ec="k"),"coil_1")
        for i in range(2,8):
            z_i = 1.173+.278 + (i-2)*.214 - zc
            coil_i = CoilPack(r0=.381,z0=z_i,nr=12,nz=8,dr=0.0127,dz=0.0127,fc=".35",ec="k")
            self.add_component(coil_i,"coil_{0}".format(i))
        self.add_component(CoilPack(r0=.286,z0=2.811-zc,nr=16,nz=16,dr=0.0135,dz=0.0135,fc=".35",ec="k"),"coil_8")
#        self.add_component(CoilPack(r0=.53,z0=3.3,nr=3,nz=6,dr=0.01,dz=0.01,fc=".35",ec="k"),"coil_9")
#        self.add_component(CoilPack(r0=.53,z0=5.3,nr=3,nz=6,dr=0.01,dz=0.01,fc=".35",ec="k"),"coil_10")

        self.grid = kwargs.pop("grid",None)
