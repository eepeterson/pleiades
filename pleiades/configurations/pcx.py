#import numpy as np
#import os
#import warnings
#from matplotlib.patches import Polygon, Wedge
#from matplotlib.collections import PatchCollection
#import matplotlib as mpl
#
#from pleiades import (Current, CurrentGroup, Magnet, Component, ZSymmCoilSet,
#                      MagnetGroup, CurrentArray, Configuration)
#
#
#class PCX_HH(object):
#    """PCX Helmholtz coil set
#
#    Attributes:
#        t_current (double): current through top HH coil
#        b_current (double): current through bottom HH coil
#        fc (str): facecolor for patch
#        top_coil (CurrentArray): CurrentArray object for top coil
#        bot_coil (CurrentArray): CurrentArray object for bottom coil
#        current_objs (list): list of Current objects comprising this instance
#        patches (list of matplotlib.patches.Polygon instances): patches representing the PCX HH coils
#        """
#
#    def __init__(self, t_current, b_current, fc='0.35'):
#        ## Build PCX HH coils
#        N=89. #number of windings (guess...)
#        self.t_current = t_current * N
#        self.b_current = b_current * N
#        self.fc = fc
#        R = 75.8825/100.
#        ztop = 38.03142/100.
#        zbot = -37.85616/100.
#        w = 10.795/100.
#        h = 10.16/100.
#        self.top_coil = Current((R, ztop), self.t_current, frame="rhoz", units="m")
#        self.bot_coil = Current((R, zbot), self.b_current, frame="rhoz", units="m")
#        self.current_objs = [self.top_coil, self.bot_coil]
#        top_coil_patch = Polygon([(R-w/2.,ztop-h/2.),(R-w/2., ztop+h/2.), (R+w/2., ztop+h/2.), (R+w/2., ztop-h/2.)],
#                                 closed=True, fc=self.fc, ec='k')
#        bot_coil_patch = Polygon([(R-w/2.,zbot-h/2.),(R-w/2., zbot+h/2.), (R+w/2., zbot+h/2.), (R+w/2., zbot-h/2.)],
#                                 closed=True, fc=self.fc, ec='k')
#        self.patches = [top_coil_patch, bot_coil_patch]
#
#    def get_current_objs(self):
#        return self.current_objs
#
#    def get_current_tuples(self, frame='rhoz', units='m'):
#        assert frame.lower() in ['polar', 'rhoz'], "Invalid frame choice: {0}".format(frame)
#        assert units.lower() in ['m', 'cm'], "Invalid units choice: {0}".format(units)
#        return [c_obj.get_current_tuples(frame=frame, units=units)[0] for c_obj in self.current_objs]
#
#
#class PCX_magCage(object):
#    """Represent an array of dipole magnets that comprise the PCX magnet cage.
#
#    Attributes:
#        magnets (list): list of Magnet objects comprising this instance
#        patches (list of matplotlib.patches.Polygon instances): patches representing the cage magnets
#    """
#
#    def __init__(self, current_mags=None):
#        ### Build the magnet array ###
#        height = 1.905  # cm
#        width = 1.905  # cm
#        # all positions relative to origin of the vessel, ref: gdrive sheet
#        # [TS1,TS2,TS3,TS4,TS5,TS6,TS7,TS8,TA,S14,S13,S12,S11,S10,S9,S8,S7,
#        #       S6,S5,S4,S3,S2,S1,BA,BS8,BS7,BS6,BS5,BS4,BS3,BS2,BS1]
#        R = np.array([4.1275, 9.8425, 15.5575, 21.2725, 26.9875, 32.7025, 38.4175, 44.1325, 46.6598, 46.25975, 46.25975,
#                      46.25975,46.25975, 46.25975, 46.25975, 46.25975, 46.25975, 46.25975, 46.25975, 46.25975, 46.25975,
#                      46.25975,46.25975, 46.6598, 44.1325, 38.4175, 32.7025, 26.9875, 21.2725, 15.5575, 9.8425, 4.1275])
#        Z = np.array([50.2335, 50.2335, 50.2335, 50.2335, 50.2335, 50.2335, 50.2335, 50.2335,49.0512,46.2645,40.0959,
#                      33.9273,27.7587,21.5901,15.4215,9.2529,3.0843,-3.0843,-9.2529,-15.4215,-21.5901,-27.7587,-33.9273,
#                      -36.7139,-37.8965,-37.8965,-37.8965,-37.8965,-37.8965,-37.8965,-37.8965,-37.8965])
#        muHats = np.array([180.,0.,180.,0.,180.,0.,180.,0.,135.,270.,90.,270.,90.,270.,90.,270.,90.,270.,90.,270.,90.,270.,
#                      90.,225.,0.,180.,0.,180.,0.,180.,0.,180.])
#        if current_mags == None:
#            current_mags = np.ones(10)
#            n = len(current_mags) / (height / 100.0)
#            ## MPDX strengths
#            current_mags *= .4 / (4 * np.pi * 10 ** -7 * n)
#            current_mags *= 3.3527
#        self.magnets = []
#        self.patches = []
#        for i, (r, z, h) in enumerate(zip(R, Z, muHats)):
#            if np.mod(i,2):
#                fc = "b"
#            else:
#                fc = "r"
#            m = Magnet((r, z), current_mags=current_mags, width=width, height=height, frame='rhoz', units="cm",
#                       mu_hat=h, fc= fc)
#            self.magnets.append(m)
#            self.patches.append(m.patch)
#
#    def set_strength(self, current_mags):
#        """Set strength of each magnet with 1D array current_mags"""
#        for m in self.magnets:
#            m.set_currents(current_mags)
#
#    def get_current_tuples(self, frame='rhoz', units='m'):
#        """Return computationally relevant info: list of (rho, z, current) tuples for instance."""
#        assert frame.lower() in ["polar", "rhoz"], "Invalid frame choice: {0}".format(frame)
#        assert units.lower() in ["m", "cm"], "Invalid units choice: {0}".format(units)
#        return [c_obj.get_current_tuples(frame=frame, units=units)[0] for c_obj in self.magnets]
#
#    def set_magnets(self, current_mags):
#        self.magnets = []
#        self.patches = []
#        self.current_mags = current_mags
#        for i, (r, zz, h) in enumerate(zip(self.r, self.z, self.muHats)):
#            if np.mod(i, 2):
#                fc = "r"
#            else:
#                fc = "b"
#            m = Magnet((r, zz), current_mags=current_mags, width=width, height=height, frame='rhoz',
#                       units='cm', mu_hat=h, fc=fc)
#            self.magnets.append(m)
#            self.patches.append(m.patch)
#
#    def set_strength(self, current_mags):
#        """Set strength of each magnet with 1D array current_mags"""
#        self.current_mags = current_mags
#        for m in self.magnets:
#            m.set_currents(current_mags)
