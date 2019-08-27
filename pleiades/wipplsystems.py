from __future__ import print_function, division, absolute_import, unicode_literals
from pleiades.core import *
import pleiades.grids
import numpy as np
from matplotlib.patches import Polygon,Wedge,Rectangle
from matplotlib.collections import PatchCollection
import matplotlib as mpl
import os
import warnings

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
    
class PCXCoil(CurrentGroup):
    def __init__(self,**kwargs):
        z0 = float(kwargs.pop("z0",0))
        self._z0 = z0
        Rarr = 0.759-7.5*0.00635 + np.arange(16)*0.00635
        Zarr = z0-(2.5*0.00952+3*0.00635) + np.concatenate((np.array([0]),np.cumsum(np.tile([0.00635,0.00952],6)[0:-1])))
        rz_pts = []
        for r in Rarr:
            for z in Zarr:
                rz_pts.append((r,z))
        rz_pts = np.array(rz_pts)
        super_kwargs = {"rz_pts":rz_pts,"patchcls":Polygon,"fc":".35","ec":"k"}
        super_kwargs.update(kwargs)
        super(PCXCoil,self).__init__(**super_kwargs)
        
    @property
    def z0(self):
        return self._z0

    @z0.setter
    def z0(self,new_z0):
        dz = new_z0 - self._z0
        super(PCXCoil,self).translate((0,dz))
        self._z0 = new_z0
    
    def build_patchargs(self,**kwargs):
        z0 = self._z0
        left,right = 0.759-7.5*0.00635,0.759+7.5*0.00635
        bottom,top = z0 - (2.5*0.00952+3*0.00635), z0 + (2.5*0.00952+3*0.00635)
        return (np.array([[left,bottom],[left,top],[right,top],[right,bottom]]),)
        

class TREXCoils(Component):
    def __init__(self,**kwargs):
        ###### Build TREX coils
        super(TREXCoils,self).__init__()
        z0 = float(kwargs.pop("z0",1.1757))
        labels = kwargs.pop("labels",["Ncoil","Scoil"])
        currents = array(kwargs.pop("currents",(1,1)),dtype="float")
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
        
class PCXCoils(Component):
    def __init__(self,**kwargs):
        super(PCXCoils,self).__init__()
        z0 = float(kwargs.pop("z0",0.375))
        labels = kwargs.pop("labels",["TopCoil","BotCoil"])
        currents = array(kwargs.pop("currents",(1,1)),dtype="float")
        nprocs = kwargs.pop("nprocs",[4,4])
        patch_mask = kwargs.pop("patch_mask",[0,0])
        grid = kwargs.pop("grid",None)
        TopCoil = PCXCoil(z0=z0,**kwargs)
        BotCoil = PCXCoil(z0=-z0,**kwargs)
        self.groups = [TopCoil,BotCoil]
        self.labels = labels
        self.currents = currents
        self.nprocs = nprocs
        self.patch_mask = patch_mask

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

class PCXMagnets(Component):
    def __init__(self,**kwargs):
        super(PCXMagnets,self).__init__()
        labels = kwargs.pop("labels",["TopSpider","Side","Ang","BotSpider"])
        nprocs = kwargs.pop("nprocs",[4,7,1,4])
        currents = kwargs.pop("currents",[1479.54]*4) #Calibrated by Ken 4/2/2019
        patch_mask = kwargs.pop("patch_mask",[0,0,0,0])
        height = 0.01905
        width = 0.01905
        ## Top Spider Group ##
        zpts = 0.50233413*np.ones(8)
        rpts = np.arange(0.041275,0.45,0.05715)
        rz_pts = np.vstack((rpts,zpts)).T
        mu_hats = np.tile([180,0],4)
        mTS = MagnetGroup(rz_pts=rz_pts,mu_hats=mu_hats,height=height,width=width,**kwargs)
        for m_obj in mTS.obj_list:
            m_obj.patchkwargs["fc"]=".35"
        ## Side Group ##
        rpts = 0.4625975*np.ones(14)
        zpts = np.linspace(-0.33927542,0.47,0.06168644)
        rz_pts = np.vstack((rpts,zpts)).T
        mu_hats = np.tile([270,90],7)
        mS = MagnetGroup(rz_pts=rz_pts,mu_hats=mu_hats,height=height,width=width,**kwargs)
        for m_obj in mS.obj_list:
            m_obj.patchkwargs["fc"]=".35"
        ## Angle Group ##
        rpts = 0.46659800 * np.ones(2)
        zpts = np.array([-0.36713922,0.4905121])
        mu_hats = np.array([135,225])
        rz_pts = np.vstack((rpts,zpts)).T
        mAN = MagnetGroup(rz_pts=rz_pts,mu_hats=mu_hats,height=height,width=width,**kwargs)
        mAN.obj_list[0].patchkwargs["fc"] = ".35"
        mAN.obj_list[1].patchkwargs["fc"] = ".35"
        ## Bottom Spider Group ##
        zpts = -0.37896125*np.ones(8)
        rpts = np.arange(0.041275,0.45,0.05715)
        rz_pts = np.vstack((rpts,zpts)).T
        mu_hats = np.tile([180,0],4)
        mBS = MagnetGroup(rz_pts=rz_pts,mu_hats=mu_hats,height=height,width=width,**kwargs)
        for m_obj in mBS.obj_list:
            m_obj.patchkwargs["fc"]=".35"
        #################
        self.groups = [mTS,mS,mAN,mBS]
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
        
class PCX(Configuration):
    def __init__(self,**kwargs):
        super(PCX,self).__init__()
        self.add_component(PCXCoils(),"HH")
        self.add_component(PCXMagnets(),"mags")
        self.grid = kwargs.pop("grid",None)
        self.artists = [Rectangle((0.502,-0.454),.006,0.926,ec="None",fc=".35",zorder=100),
                       Rectangle((0.502,0.472),0.057,0.178,ec="None",fc=".35",zorder=100),
                       Wedge((0,0.3747),0.972,270,301.51,0.006,ec="None",fc=".35",zorder=100),
                       Wedge((0,0.026),0.785,45,90,0.006,ec="None",fc=".35",zorder=100)]
        
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
        self.coil = RectCurrentArray((rho0,z0-z1),nrho,nz,delta_rho,delta_z,current,fc=".45",units="m")
        self.patches = [self.cryo,self.coil.patch]

    def get_current_tuples(self,frame="rhoz",units="m"):
        return self.coil.get_current_tuples(frame=frame,units=units)

def build_gdt():
    rho0 = 40
    n_rho = 20
    n_z = 100
    delta_rho = 1
    delta_z = 1
    coil_1 = RectCurrentArray((rho0, -325), n_rho, n_z, delta_rho, delta_z, 1500, units="cm")
    coil_2 = RectCurrentArray((rho0, -215), n_rho, n_z, delta_rho, delta_z, 500, units="cm")
    coil_3 = RectCurrentArray((rho0, -105), n_rho, n_z, delta_rho, delta_z, 500, units="cm")
    coil_4 = RectCurrentArray((rho0, 5), n_rho, n_z, delta_rho, delta_z, 500, units="cm")
    coil_5 = RectCurrentArray((rho0, 115), n_rho, n_z, delta_rho, delta_z, 500, units="cm")
    coil_6 = RectCurrentArray((rho0, 225), n_rho, n_z, delta_rho, delta_z, 1500, units="cm")
    coil_7 = RectCurrentArray((rho0 / 2.0, -395), 10, 50, delta_rho, delta_z, 30000, units="cm")
    coil_8 = RectCurrentArray((rho0 / 2.0, 345), 10, 50, delta_rho, delta_z, 30000, units="cm")
    current_objs = [coil_1, coil_2, coil_3, coil_4, coil_5, coil_6, coil_7, coil_8]
    patches = [coil_1.patch, coil_2.patch, coil_3.patch, coil_4.patch, coil_5.patch, coil_6.patch, coil_7.patch,
               coil_8.patch]

    return current_objs, patches
