from numpy import (pi, linspace, meshgrid, sin, cos, sqrt, sum, array, ones,
                   zeros, hstack, vstack, sign, mod, isfinite, ceil, isclose)
import numpy as np
from scipy.special import ellipk, ellipe
from multiprocessing import Process, Queue, cpu_count
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.transforms import Affine2D
from numbers import Number
from warnings import warn, simplefilter
import pickle
from abc import ABCMeta


class CurrentFilamentSet(metaclass=ABCMeta):
    """Set of locations that have the same current value

    Parameters
    ----------
    rz_pts : iterable
        Nx2 iterable representing R,Z current centroids. Defaults to None
    current : float, optional
        The current to be used for Green's function.
    weights : iterable, optional
        The weights for all the current locations. This enables having both
        positive and negative currents in an object at the same time as well as
        current profile shaping in the case of permanent magnets. Defaults to
        1 for every location

    Attributes
    ----------
    rz_pts : iterable
        Nx2 iterable representing R,Z current centroids. Defaults to None
    current : float
        The current in the CurrentGroup in amps.
    weights : iterable
        The weights for all the current locations. This enables having both
        positive and negative currents in an object at the same time as well as
        current profile shaping in the case of permanent magnets. Defaults to
        1 for every location
    rzw : np.array
        An Nx3 array whos rows are rzw[i, :] = rloc, zloc, weight which
        describe the current location and current weight for each filament in
        the CurrentFilamentSet. This is simply a helper attribute for combining
        rz_pts and weights.
    markers: iterable of str
        A list of strings in {'x', 'o'} denoting current into and out of the
        page which are calculated by the sign of the current times the weight
        for each filament location.
    g_psi : GreensFunction object
        An object representing the Green's function for magnetic flux for this
        CurrentFilamentSet.
    g_br : GreensFunction object
        An object representing the Green's function for the radial component B_R
        for this CurrentFilamentSet.
    g_bz : GreensFunction object
        An object representing the Green's function for the radial component B_R
        for this CurrentFilamentSet.
    """

    def __init__(self, rz_pts, current=1., weights=None):
        self._rz_pts = None
        self.rz_pts = rz_pts
        self.current = current
        self.weights = weights

    @property
    def rz_pts(self):
        return self._rz_pts

    @property
    def current(self):
        return self._current

    @property
    def weights(self):
        return self._weights

    @property
    def rzw(self):
        return np.hstack(self._rz_pts, self._weights)

    @property
    def markers(self):
        cur = self.current
        return ['x' if cur*w > 0 else 'o' for w in self._weights]

    @rz_pts.setter
    def rz_pts(self, rz_pts):
        if self._rz_pts is not None and len(rz_pts) != len(self._rz_pts):
            warn('Resetting weights to ones since current filament number was '
                 'not preserved by this operation', UserWarning)
            self._weights = np.ones(len(rz_pts))
        self._rz_pts = np.asarray(rz_pts)

    @current.setter
    def current(self, current):
        self._current = current

    @weights.setter
    def weights(self, weights):
        if weights is None:
            self._weights = np.ones(len(self._rz_pts))
        else:
            assert len(weights) == len(self._rz_pts)
            self._weights = np.asarray(weights)

    def translate(self, dr, dz):
        """Translate the current group by the vector (dr, dz)

        Parameters
        ----------
        dr : float
            The displacement in the R direction for the translation
        dz : float
            The displacement in the Z direction for the translation
        """
        self.rz_pts = self.rz_pts + np.array([dr, dz]).reshape((1, 2))

    def rotate(self, angle, r0=0., z0=0.):
        """Rotate the current group by a given angle around a specified pivot

        Parameters
        ----------
        angle : float
            The angle of the rotation in degrees as measured from the z axis
        r0 : float, optional
            The R location of the pivot. Defaults to 0.
        z0 : float, optional
            The Z location of the pivot. Defaults to 0.
        """
        angle = pi / 180.0 * angle
        pivot = np.array([r0, z0]).reshape((1, 2))
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        self.rz_pts = (self.rz_pts - pivot) @ rotation + pivot

    def plot_currents(self, ax, **kwargs):
        """Plot the current locations for the CurrentGroup

        Parameters
        ----------
        ax : matplotlib.Axes object
            The axes object for plotting the current locations
        **kwargs : dict, optional
            Keyword arguments to pass to Current.plot method
        """
        markers = self.markers
        for i, (r, z, w) in enumerate(self.rzw):
            ax.plot(r, z, markers[i], **kwargs)

    def psi(self, grid):
        """Compute the magnetic flux, psi, on the desired grid.

        Parameters
        ----------
        grid : np.array
            (R, Z) points at which to calculate the magnetic flux

        Returns
        -------
        psi : np.array
        """
        return self.current*self.g_psi(grid)

    def br(self, grid):
        """Compute the radial component of the magnetic field, B_R, on the
        desired grid.

        Parameters
        ----------
        grid : np.array
            (R, Z) points at which to calculate the magnetic field Br

        Returns
        -------
        br : np.array
        """
        return self.current*self.g_br(grid)

    def bz(self, grid):
        """Compute the radial component of the magnetic field, B_Z, on the
        desired grid.

        Parameters
        ----------
        grid : np.array
            (R, Z) points at which to calculate the magnetic field Bz

        Returns
        -------
        bz : np.array
        """
        return self.current*self.g_br(grid)


class RectangularCoil(CurrentFilamentSet):
    """A rectangular cross section coil in the R-Z plane

    Parameters
    ----------
    r0 : float
        The R location of the centroid of the coil
    z0 : float
        The Z location of the centroid of the coil
    nr : float
        The number of current filaments in the R direction. Defaults to 10.
    nz : float
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float
        The distance between current filaments in the Z direction. Defaults to
        0.01 m

    Attributes
    ----------
    r0 : float
        The R location of the centroid of the Coil
    z0 : float
        The Z location of the centroid of the Coil
    nr : float
        The number of current filaments in the R direction. Defaults to 10.
    nz : float
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float
        The distance between current filaments in the R direction. Defaults to
        0.1 m
    dz : float
        The distance between current filaments in the Z direction. Defaults to
        0.1 m
    """
    def __init__(self, r0, z0, nr=1, nz=1, dr=0.1, dz=0.1, **kwargs):
        self._r0 = r0
        self._z0 = z0
        self._nr = nr
        self._nz = nz
        self._dr = dr
        self._dz = dz
        super().__init__(self._compute_rz_pts(), **kwargs)

    @property
    def r0(self):
        return self._r0

    @property
    def z0(self):
        return self._z0

    @property
    def nr(self):
        return self._nr

    @property
    def nz(self):
        return self._nz

    @property
    def dr(self):
        return self._dr

    @property
    def dz(self):
        return self._dz

    def _compute_rz_pts(self):
        """Compute the rz_pts locations from this coil's internal parameters"""
        r0, nr, dr = self._r0, self._nr, self._dr
        z0, nz, dz = self._z0, self._nz, self._dz
        rl, ru = r0 - dr*(nr - 1)/2, r0 + dr*(nr - 1)/2
        zl, zu = z0 - dz*(nz - 1)/2, z0 + dz*(nz - 1)/2
        r = np.linspace(rl, ru, nr)
        z = np.linspace(zl, zu, nz)
        rr, zz = np.meshgrid(r, z)
        return np.vstack((rr.ravel(), zz.ravel())).T


class MagnetRing(CurrentFilamentSet):
    """Represent a Rectangular cross-section dipole magnet with axisymmetric 
    surface currents.

    Parameters
    ----------
    rz_pts : iterable, optional
        Nx2 iterable representing R,Z current centroids. Defaults to None
    current : float, optional
        The current in all the current ring in amps, defaults to 1 amp.
    kwargs : matplotlib patch keyword arguments

    Attributes
    ----------
    loc : tuple
        The (R, Z) location of the centroid of the magnet.
    width : float, optional
        The width of the magnet. Defaults to 0.01 m.
    height : float, optional
        The height of the magnet. Defaults to 0.01 m.
    mu_hat : float, optional
        The angle of the magnetic moment of the magnet in degrees from the z
        axis. Defaults to 0 degrees clockwise from Z axis (i.e. north pole
        points in the +z direction).
    current_prof : integer or array_like
        The current profile along the side of the magnet. Defaults to
        np.ones(8) i.e. 8 equal surface currents per side.
    current : float
        The current in the magnet in amps.
    obj_list : list
        The list of Current objects that comprise the Magnet
    rzdir : np.array
        An Nx3 array whos rows are rzdir[i, :] = rloc, zloc, current which
        describe the current location and current value for each current in the
        Magnet
    patch : matplotlib.patches.Patch object
        The patch object representing the Magnet for plotting
    patchkwargs : dict
        The keyword arguments used for the patch attribute
    """

    def __init__(self, **kwargs):
        Magnet.reset(self, **kwargs)

    def reset(self, **kwargs):
        # set Magnet specific attributes before calling super constructor
        r0, z0 = kwargs.pop("loc", (1.0, 1.0))
        if r0 < 0:
            raise ValueError("Centroid of magnet, r0, must be >= 0")
        r0 = float(r0)
        z0 = float(z0)
        width = float(kwargs.pop("width", .01))
        height = float(kwargs.pop("height", .01))
        if not (width > 0 and height > 0):
            raise ValueError("width and height must be greater than 0")
        self._width = width
        self._height = height
        ## need to pop this now but save it for later
        mu_hat = kwargs.pop("mu_hat", 0)
        self._mu_hat = 0
        current_prof = kwargs.pop("current_prof", 10)
        if isinstance(current_prof, Number):
            current_prof = ones(current_prof)
        else:
            current_prof = array(current_prof)
        if not current_prof.size > 0:
            raise ValueError("current_prof array must have size > 0")
        self._current_prof = current_prof
        self._loc = (r0, z0)
        # start building super class relevant inputs
        # super_kwargs include rz_pts,current,patchcls,patchargs_dict, any matplotlib.patches kwarg
        current = kwargs.pop("current", 1)
        if not current > 0:
            raise ValueError("current must be > 0")
        self._current = current
        n = len(self._current_prof)
        dummy = ones(n)
        rpts = self._width / 2.0 * hstack((-1 * dummy, dummy))
        if n == 1:
            zpts = zeros(2)
        else:
            ztmp = linspace(-self._height / 2.0, self._height / 2.0, n)
            zpts = hstack((ztmp, ztmp))
        rz_pts = vstack((rpts + r0, zpts + z0)).T
        patchkwargs = {"closed": True, "fc": "w", "ec": "k", "zorder": 3}
        # All leftover kwargs get put into patchkwargs
        patchkwargs.update(kwargs)
        # Build kwargs for super constructor
        super_kwargs = {"rz_pts": rz_pts, "current": 1.0, "patchcls": Polygon, "patchargs_dict": {}}
        super_kwargs.update(patchkwargs)
        # builds CurrentGroup at loc with current = 1 for all current objs
        super(Magnet, self).__init__(**super_kwargs)
        # make left side currents negative (current setter overridden below)
        self.current = self._current
        # rotate according to muhat direction
        self.mu_hat = mu_hat

    @property
    def loc(self):
        return self._loc

    @loc.setter
    def loc(self, r0, z0):
        self.rebuild("loc", (r0, z0))

    @CurrentGroup.current.setter
    def current(self, new_current):
        # makes first half of obj_list have negative currents
        if new_current < 0:
            raise ValueError("current for Magnet class must be > 0")
        self._current = new_current
        n = len(self._obj_list) / 2
        for i, c_obj in enumerate(self._obj_list):
            c_obj.current = new_current * (-1) ** (i // n + 1)

    @property
    def width(self):
        return self._width

    @width.setter
    def width(self, new_width):
        self.rebuild("width", new_width)

    @property
    def height(self):
        return self._height

    @height.setter
    def height(self, new_height):
        self.rebuild("height", new_height)

    @property
    def current_prof(self):
        return self._current_prof

    @current_prof.setter
    def current_prof(self, new_prof):
        self.rebuild("current_prof", new_prof)

    @property
    def mu_hat(self):
        return self._mu_hat

    @mu_hat.setter
    def mu_hat(self, mu_hat):
        self.rotate(mu_hat - self._mu_hat)


class CurrentFilamentMultiSet(metaclass=ABCMeta):
    """A helper class for storing multiple CurrentFilamentSets in one object
    each with their own current value and Green's function.
    """
    pass


