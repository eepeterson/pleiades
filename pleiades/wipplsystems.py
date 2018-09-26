from __future__ import print_function, division, absolute_import, unicode_literals
from pleiades.core import *
import pleiades.grids
import numpy as np
from matplotlib.patches import Polygon,Wedge
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
        plist = [group.patches for group,mask in izip(self._groups,self._patch_mask) if not mask]
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
        plist = [group.patches for group,mask in izip(self._groups,self._patch_mask) if not mask]
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



class PCX_HH(object):
    """PCX Helmholtz coil set

    Attributes:
        t_current (double): current through top HH coil
        b_current (double): current through bottom HH coil
        fc (str): facecolor for patch
        top_coil (CurrentArray): CurrentArray object for top coil
        bot_coil (CurrentArray): CurrentArray object for bottom coil
        current_objs (list): list of Current objects comprising this instance
        patches (list of matplotlib.patches.Polygon instances): patches representing the PCX HH coils
        """

    def __init__(self, t_current, b_current, fc='0.35'):
        ## Build PCX HH coils
        N=89. #number of windings (guess...)
        self.t_current = t_current * N
        self.b_current = b_current * N
        self.fc = fc
        R = 75.8825/100.
        ztop = 38.03142/100.
        zbot = -37.85616/100.
        w = 10.795/100.
        h = 10.16/100.
        self.top_coil = Current((R, ztop), self.t_current, frame="rhoz", units="m")
        self.bot_coil = Current((R, zbot), self.b_current, frame="rhoz", units="m")
        self.current_objs = [self.top_coil, self.bot_coil]
        top_coil_patch = Polygon([(R-w/2.,ztop-h/2.),(R-w/2., ztop+h/2.), (R+w/2., ztop+h/2.), (R+w/2., ztop-h/2.)],
                                 closed=True, fc=self.fc, ec='k')
        bot_coil_patch = Polygon([(R-w/2.,zbot-h/2.),(R-w/2., zbot+h/2.), (R+w/2., zbot+h/2.), (R+w/2., zbot-h/2.)],
                                 closed=True, fc=self.fc, ec='k')
        self.patches = [top_coil_patch, bot_coil_patch]

    def get_current_objs(self):
        return self.current_objs

    def get_current_tuples(self, frame='rhoz', units='m'):
        assert frame.lower() in ['polar', 'rhoz'], "Invalid frame choice: {0}".format(frame)
        assert units.lower() in ['m', 'cm'], "Invalid units choice: {0}".format(units)
        return [c_obj.get_current_tuples(frame=frame, units=units)[0] for c_obj in self.current_objs]


class PCX_magCage(object):
    """Represent an array of dipole magnets that comprise the PCX magnet cage.

    Attributes:
        magnets (list): list of Magnet objects comprising this instance
        patches (list of matplotlib.patches.Polygon instances): patches representing the cage magnets
    """

    def __init__(self, current_mags=None):
        ### Build the magnet array ###
        height = 1.905  # cm
        width = 1.905  # cm
        # all positions relative to origin of the vessel, ref: gdrive sheet
        # [TS1,TS2,TS3,TS4,TS5,TS6,TS7,TS8,TA,S14,S13,S12,S11,S10,S9,S8,S7,
        #       S6,S5,S4,S3,S2,S1,BA,BS8,BS7,BS6,BS5,BS4,BS3,BS2,BS1]
        R = np.array([4.1275, 9.8425, 15.5575, 21.2725, 26.9875, 32.7025, 38.4175, 44.1325, 46.6598, 46.25975, 46.25975,
                      46.25975,46.25975, 46.25975, 46.25975, 46.25975, 46.25975, 46.25975, 46.25975, 46.25975, 46.25975,
                      46.25975,46.25975, 46.6598, 44.1325, 38.4175, 32.7025, 26.9875, 21.2725, 15.5575, 9.8425, 4.1275])
        Z = np.array([50.2335, 50.2335, 50.2335, 50.2335, 50.2335, 50.2335, 50.2335, 50.2335,49.0512,46.2645,40.0959,
                      33.9273,27.7587,21.5901,15.4215,9.2529,3.0843,-3.0843,-9.2529,-15.4215,-21.5901,-27.7587,-33.9273,
                      -36.7139,-37.8965,-37.8965,-37.8965,-37.8965,-37.8965,-37.8965,-37.8965,-37.8965])
        muHats = np.array([180.,0.,180.,0.,180.,0.,180.,0.,135.,270.,90.,270.,90.,270.,90.,270.,90.,270.,90.,270.,90.,270.,
                      90.,225.,0.,180.,0.,180.,0.,180.,0.,180.])
        if current_mags == None:
            current_mags = np.ones(10)
            n = len(current_mags) / (height / 100.0)
            ## MPDX strengths
            current_mags *= .4 / (4 * np.pi * 10 ** -7 * n)
            current_mags *= 3.3527
        self.magnets = []
        self.patches = []
        for i, (r, z, h) in enumerate(zip(R, Z, muHats)):
            if np.mod(i,2):
                fc = "b"
            else:
                fc = "r"
            m = Magnet((r, z), current_mags=current_mags, width=width, height=height, frame='rhoz', units="cm",
                       mu_hat=h, fc= fc)
            self.magnets.append(m)
            self.patches.append(m.patch)

    def set_strength(self, current_mags):
        """Set strength of each magnet with 1D array current_mags"""
        for m in self.magnets:
            m.set_currents(current_mags)

    def get_current_tuples(self, frame='rhoz', units='m'):
        """Return computationally relevant info: list of (rho, z, current) tuples for instance."""
        assert frame.lower() in ["polar", "rhoz"], "Invalid frame choice: {0}".format(frame)
        assert units.lower() in ["m", "cm"], "Invalid units choice: {0}".format(units)
        return [c_obj.get_current_tuples(frame=frame, units=units)[0] for c_obj in self.magnets]

    def set_magnets(self, current_mags):
        self.magnets = []
        self.patches = []
        self.current_mags = current_mags
        for i, (r, zz, h) in enumerate(zip(self.r, self.z, self.muHats)):
            if np.mod(i, 2):
                fc = "r"
            else:
                fc = "b"
            m = Magnet((r, zz), current_mags=current_mags, width=width, height=height, frame='rhoz',
                       units='cm', mu_hat=h, fc=fc)
            self.magnets.append(m)
            self.patches.append(m.patch)

    def set_strength(self, current_mags):
        """Set strength of each magnet with 1D array current_mags"""
        self.current_mags = current_mags
        for m in self.magnets:
            m.set_currents(current_mags)




class PhilipsMRI(object):
    def __init__(self, (rho0, z0), current):
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

def build_pcx(vessel=True, HH=(False, 0, 0)):
    """"Return field objects and patches representing PCX (modified copy of build_wipal).

    Kwargs:
        vessel (bool): boolean to include vessel magnets or not, default True
        HH (tuple): tuple of (bool,float,float) representing whether or not to include
            helmholtz coil set and, if so, how much current goes into the upper and lower coil, respectively
            default (False,0,0)
    """
    patches = []
    current_objs = []
    if vessel:
        vessel_magnets = PCX_magCage()
        current_objs.extend(vessel_magnets.magnets)
        patches += vessel_magnets.patches
    if HH[0]:
        top_current = HH[1]
        bot_current = HH[2]
        hh_coils = PCX_HH(top_current, bot_current)
        current_objs.extend(hh_coils.get_current_objs())
        patches.extend(hh_coils.patches)
    return current_objs, patches

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
