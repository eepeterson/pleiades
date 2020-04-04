from abc import ABCMeta, abstractmethod, abstractproperty
from collections import Iterable
from warnings import warn, simplefilter
import math
import numpy as np
from scipy.special import ellipk, ellipe
from matplotlib.path import Path
import matplotlib.patches as patches

from pleiades import Grid, compute_greens


def rotate(pts, angle, pivot=(0., 0.)):
    pivot = np.asarray(pivot)
    angle = math.pi*angle/180
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return (np.asarray(pts) - pivot) @ rotation + pivot


def check_greens_on_get(func):
    def wrapper(self):
        if not self._uptodate:
            self._compute_greens()
        return func(self)
    return wrapper


def check_greens_on_set(name):
    def actual_decorator(func):
        def wrapper(self, value):
            self._flag_greens(getattr(self, name), value)
            return func(self, value)
        return wrapper
    return actual_decorator


class CurrentFilamentSet(metaclass=ABCMeta):
    """Set of locations that have the same current value.

    A CurrentFilamentSet represents a set of axisymmetric current centroids with
    associated current weights to describe the current ratios between all the
    centroids. In addition a CurrentFilamentSet implements the Green's function
    functionality for computing magnetic fields and flux on an R-Z grid. A
    CurrentFilamentSet is not intended to be instatiated directly, but serves as
    the base class for all concrete current set classes and defines the
    minimum functional interface and protocols of a current set. Lastly a
    matplotlib.patches.PathPatch object is associated with each
    CurrentFilamentSet for ease in plotting and verifying device geometry.

    Parameters
    ----------
    current : float, optional
        The current to be used for Green's function.
    weights : iterable, optional
        The weights for all the current locations. This enables having both
        positive and negative currents in an object at the same time as well as
        current profile shaping in the case of permanent magnets. Defaults to
        1 for every location.
    **kwargs
        Any matplotlib.patches.Patch keyword arguments. No type checking is
        performed on these inputs they are simply assigned to the patch_kw
        attribute.

    Attributes
    ----------
    current : float
        The current in the CurrentGroup in amps.
    weights : iterable
        The weights for all the current locations. This enables having both
        positive and negative currents in an object at the same time as well as
        current profile shaping in the case of permanent magnets. Defaults to
        1 for every location
    npts : int
        Integer for the number of current filaments in this object.
    rz_pts : np.ndarray
        Nx2 array representing R, Z current centroids.
    rzw : np.ndarray
        An Nx3 array whos rows are rzw[i, :] = rloc, zloc, weight which
        describe the current location and current weight for each filament in
        the CurrentFilamentSet. This is simply a helper attribute for combining
        rz_pts and weights.
    total_current : float
        The total current being carried in the filament set. This is equal to
        the current times the sum of the weights.
    """

    _EPS = 1E-12
    _SET_GREENS_TRIGS = ['weights', 'grid']

    def __init__(self, current=1., weights=None, **kwargs):
        self._current = None
        self._weights = None
        self._grid = None
        self._grid_pts = None

        self.current = current
        self.weights = weights
        self.patch_kw = kwargs

#    def __setattr__(self, name, value):
#        if name in type(self)._SET_GREENS_TRIGS:
#            self._flag_greens(getattr(self, name), value)
#        return super().__setattr__(name, value)

    @abstractproperty
    def npts(self):
        pass

    @abstractproperty
    def rz_pts(self):
        pass

    @abstractproperty
    def patch(self):
        pass

    @property
    def current(self):
        return self._current

    @property
    def weights(self):
        return self._weights

    @property
    def rzw(self):
        rzw = np.empty((self.npts, 3))
        rzw[:, 0:2] = self.rz_pts
        rzw[:, 2] = self.weights
        return rzw

    @property
    def grid(self):
        return self._grid

    @property
    @check_greens_on_get
    def gpsi(self):
        return self._gpsi

    @property
    @check_greens_on_get
    def gBR(self):
        return self._gBR

    @property
    @check_greens_on_get
    def gBZ(self):
        return self._gBZ

    @property
    def total_current(self):
        return self.current*np.sum(self.weights)

    @property
    def _markers(self):
        cw = self.current*self.weights
        cw[np.abs(cw) < self._EPS] = 0
        return ['' if cwi == 0 else 'x' if cwi > 0 else 'o' for cwi in cw]

    @current.setter
    def current(self, current):
        self._current = current

    @weights.setter
    @check_greens_on_set('weights')
    def weights(self, weights):
        if weights is None:
            self._weights = np.ones(self.npts)
        else:
            assert len(weights) == self.npts
            self._weights = np.asarray(weights)

    @grid.setter
    def grid(self, grid):
        """Parses input grid variable into Nx2 array of points.

        Checks value of current against the value of future and decides whether
        or not to set self._uptodate to True or False. This function is intended
        to make the user experience smoother while preserving performance by
        caching the Green's functions. Variables that require recomputing
        Green's functions include any positional variables or weights.

        Parameters
        ----------
        grid : 2-tuple of np.array, or Nx2 np.array, or pleiades.Grid object
            An Nx2 array of (R, Z) points, or a tuple of R, Z pts or a
            pleiades.Grid object to evaluate the Green's function on.
        """
        # Prep rz_pts array and get shape based on input 
        if isinstance(grid, Grid):
            shape = grid.R.shape
            rz_pts = np.empty((grid.R.size, 2))
            rz_pts[:, 0] = grid.R.ravel()
            rz_pts[:, 1] = grid.Z.ravel()
        elif isinstance(grid, np.ndarray):
            assert len(grid.shape) == 2
            assert grid.shape[1] == 2
            shape = grid.shape[0:1]
            rz_pts = grid
        elif isinstance(grid, Iterable):
            assert len(grid) == 2
            shape = grid[0].shape
            rz_pts = np.empty_like(grid[0])
            rz_pts[:, 0] = grid[0].ravel()
            rz_pts[:, 1] = grid[1].ravel()
        else:
            raise ValueError('Unsupported type for grid')

        self._grid = grid
        self._grid_pts = rz_pts
        self._uptodate = False

    @abstractmethod
    def translate(self, vector):
        """Translate the current group by the vector (dr, dz)

        Parameters
        ----------
        vector : iterable of float
            The displacement vector for the translation
        """

    @abstractmethod
    def rotate(self, angle, pivot=(0., 0.)):
        """Rotate the current group by a given angle around a specified pivot

        Parameters
        ----------
        angle : float
            The angle of the rotation in degrees as measured from the z axis
        pivot : iterable of float, optional
            The (R, Z) location of the pivot. Defaults to (0., 0.).
        """

    def psi(self, current=None):
        """Compute the magnetic flux, psi.

        Parameters
        ----------
        current : float, optional
            Specify a current value to override the current attribute for
            calculating the field. Defaults to None, which causes the current
            attribute to be used for the calculation

        Returns
        -------
        psi : np.array
        """
        current = self.current if current is None else current
        return current*self.gpsi

    def BR(self, current=None):
        """Compute the radial component of the magnetic field, BR.

        Parameters
        ----------
        current : float, optional
            Specify a current value to override the current attribute for
            calculating the field. Defaults to None, which causes the current
            attribute to be used for the calculation

        Returns
        -------
        BR : np.array
        """
        current = self.current if current is None else current
        return current*self.gBR

    def BZ(self, current=None):
        """Compute the z component of the magnetic field, BZ.

        Parameters
        ----------
        current : float, optional
            Specify a current value to override the current attribute for
            calculating the field. Defaults to None, which causes the current
            attribute to be used for the calculation

        Returns
        -------
        BZ : np.array
        """
        current = self.current if current is None else current
        return current*self.gBZ

    def plot(self, ax, plot_patch=True, **kwargs):
        """Plot the current locations for the CurrentGroup

        Parameters
        ----------
        ax : matplotlib.Axes object
            The axes object for plotting the current locations
        plot_patch : bool
            Whether to add the patch for this CurrentFilament to the axes.
        **kwargs : dict, optional
            Keyword arguments to pass to Current.plot method
        """
        if plot_patch:
            ax.add_patch(self.patch)

        markers = self._markers
        for i, (r, z, w) in enumerate(self.rzw):
            ax.plot(r, z, marker=markers[i], **kwargs)

    def _compute_greens(self):
        """Compute and assign the Green's functions for psi, BR, and BZ"""
        # Calculate Green's functions 
        gpsi, gBR, gBZ = compute_greens(self.rzw, self._grid_pts)
        self._gpsi = gpsi
        self._gBR = gBR
        self._gBZ = gBZ

        # Notify instance that the Green's functions are up to date
        self._uptodate = True

    def _flag_greens(self, current, future):
        """Determines if Green's functions need to be recomputed.

        Checks value of current against the value of future and decides whether
        or not to set self._uptodate to True or False. This function is intended
        to make the user experience smoother while preserving performance by
        caching the Green's functions. Variables that require recomputing
        Green's functions include any positional variables or weights.

        Parameters
        ----------
        current :
            The current value
        future :
            The future value being set
        """
        if current is None:
            self._uptodate = False
        else:
            uptodate = np.allclose(current, future, rtol=0., atol=self._EPS)
            self._uptodate = uptodate


class ArbitraryPoints(CurrentFilamentSet):
    """A set of arbitrary points in the R-Z plane with the same current

    Parameters
    ----------
    rzpts : np.ndarray
        An Nx2 array of (R,Z) locations for the filaments that comprise this
        set.
    **kwargs
        Any valid keyword arguments for CurrentFilamentSet.

    Attributes
    ----------
    """

    def __init__(self, rz_pts, **kwargs):
        self.rz_pts = np.asarray(rz_pts)
        self._angle = 0.
        super().__init__(**kwargs)

    @property
    def npts(self):
        return len(self.rz_pts)

    @property
    def rz_pts(self):
        return self._rz_pts

    @property
    def patch(self):
        return None

    @rz_pts.setter
    def rz_pts(self, rz_pts):
        self._flag_greens(rz_pts, self.rz_pts)
        self._rz_pts = np.asarray(rz_pts)
        if len(rz_pts) != len(self.weights):
            warn('Resetting weights to 1')
            self._weights = np.ones(len(rz_pts))

    def translate(self, vector):
        self._flag_greens((0., 0.), vector)
        self.rz_pts += np.array(vector)

    def rotate(self, angle, pivot=(0., 0.)):
        self._flag_greens(0., angle)
        self._angle += angle
        angle = math.pi*angle / 180
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        self.rz_pts = (self.rz_pts - pivot) @ rotation + np.asarray(pivot)


class RectangularCoil(CurrentFilamentSet):
    """A rectangular cross section coil in the R-Z plane

    Parameters
    ----------
    r0 : float
        The R location of the centroid of the coil
    z0 : float
        The Z location of the centroid of the coil
    nr : float, optional
        The number of current filaments in the R direction. Defaults to 10.
    nz : float, optional
        The number of current filaments in the Z direction. Defaults to 10.
    dr : float, optional
        The distance between current filaments in the R direction. Defaults to
        0.01 m
    dz : float, optional
        The distance between current filaments in the Z direction. Defaults to
        0.01 m
    nhat : iterable of float, optional
        A vector of (dr, dz) representing the orientation of the coil and the
        'local z direction'. This is the direction which applies to nz and dz
        when constructing current filament locations. The 'r' direction is found
        by the relation nhat x phi_hat = rhat. Defaults to (0, 1) meaning the
        local z axis is aligned with the global z axis and likewise for the r
        axis.
    **kwargs
        Any valid keyword arguments for CurrentFilamentSet.

    Attributes
    ----------
    r0 : float
        The R location of the centroid of the Coil
    z0 : float
        The Z location of the centroid of the Coil
    centroid : np.array
        Helper attribue for the R, Z location of the centroid of the Coil
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
    angle : float
        An angle in degrees representing the rotation of the coil and the 'local
        z direction' with respect to the global z axis. This is the direction
        which applies to nz and dz when constructing current filament locations.
        Defaults to 0 meaning the local z axis is aligned with the global z
        axis.
    verts : np.ndarray
        A 4x2 np.array representing the 4 vertices of the coil (read-only).
    area : float
        The area of the coil in m^2 (read-only).
    current_density : float
        The current density in the coil. This is equal to the total current
        divided by the area (read-only).
    """

    _SET_GREENS_TRIGS = CurrentFilamentSet._SET_GREENS_TRIGS
    _SET_GREENS_TRIGS += ['r0', 'z0', 'nr', 'nz', 'dr', 'dz' ,'angle']

    _codes = [Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CLOSEPOLY]

    def __init__(self, r0, z0, nr=1, nz=1, dr=0.1, dz=0.1, angle=0., **kwargs):
        self._r0 = r0
        self._z0 = z0
        self._nr = nr
        self._nz = nz
        self._dr = dr
        self._dz = dz
        self._angle = angle
        super().__init__(**kwargs)

    @property
    def r0(self):
        return self._r0

    @property
    def z0(self):
        return self._z0

    @property
    def centroid(self):
        return np.array([self.r0, self.z0])

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

    @property
    def angle(self):
        return self._angle

    @property
    def npts(self):
        return self.nr*self.nz

    @property
    def rz_pts(self):
        # Compute the rz_pts locations from this coil's internal parameters
        r0, z0 = self.centroid
        nr, dr, nz, dz = self.nr, self.dr, self.nz, self.dz
        rl, ru = r0 - dr*(nr - 1)/2, r0 + dr*(nr - 1)/2
        zl, zu = z0 - dz*(nz - 1)/2, z0 + dz*(nz - 1)/2
        r = np.linspace(rl, ru, nr)
        z = np.linspace(zl, zu, nz)
        rz_pts = np.array([(ri, zi) for ri in r for zi in z])
        if np.isclose(self.angle, 0):
            return rz_pts
        return rotate(rz_pts, self.angle, pivot=(r0, z0))

    @property
    def patch(self):
        return patches.PathPatch(Path(self._verts, self._codes))

    @property
    def _verts(self):
        # Get indices for 4 corners of current filament array
        nr, dr, nz, dz = self.nr, self.dr, self.nz, self.dz
        idx = np.array([0, nz - 1, nr*nz - 1, (nr - 1)*nz, 0])
        verts = self.rz_pts[idx, :]

        # Get correction vector to account for half width of filaments
        hdr, hdz = self.dr/2, self.dz/2
        dverts = np.array([[-hdr, -hdz],
                           [-hdr, hdz],
                           [hdr, hdz],
                           [hdr, -hdz],
                           [-hdr, -hdz]])

        if not np.isclose(self.angle, 0):
            dverts = rotate(dverts, self.angle)

        return verts + dverts

    @property
    def area(self):
        return self.nr*self.dr*self.nz*self.dz

    @property
    def current_density(self):
        return self.total_current / self.area

    @r0.setter
    @check_greens_on_set('r0')
    def r0(self, r0):
        self._r0 = r0

    @z0.setter
    def z0(self, z0):
        self._z0 = z0

    @centroid.setter
    def centroid(self, centroid):
        self.r0 = centroid[0]
        self.z0 = centroid[1]

    @nr.setter
    def nr(self, nr):
        self._nr = nr

    @nz.setter
    def nz(self, nz):
        self._nz = nz

    @dr.setter
    def dr(self, dr):
        self._dr = dr

    @dz.setter
    def dz(self, dz):
        self._dz = dz

    @angle.setter
    def angle(self, angle):
        self._angle = angle

    def translate(self, vector):
        self.centroid += np.array(vector)

    def rotate(self, angle, pivot=(0., 0.)):
        self.angle += angle
        angle = math.pi*angle / 180
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        self.centroid = (self.centroid - pivot) @ rotation + np.asarray(pivot)


class MagnetRing(CurrentFilamentSet):
    """A rectangular cross-section axisymmetric dipole magnet ring.

    A MagnetRing models a series of permanent dipole magnets placed
    axisymmetrically in a ring. The dipole magnet is modeled with anti-parallel
    surface currents on either side of the magnet.

    Parameters
    ----------
    r0 : float
        The R location of the centroid of the magnet ring
    z0 : float
        The Z location of the centroid of the magnet ring
    width : float, optional
        The width of the magnet. Defaults to 0.01 m.
    height : float, optional
        The height of the magnet. Defaults to 0.01 m.
    mu_hat : float, optional
        The angle of the magnetic moment of the magnet in degrees from the z
        axis. Defaults to 0 degrees clockwise from Z axis (i.e. north pole
        points in the +z direction).
    **kwargs
        Any valid keyword arguments for CurrentFilamentSet.

    Attributes
    ----------
    r0 : float
        The R location of the centroid of the Coil
    z0 : float
        The Z location of the centroid of the Coil
    centroid : np.array
        Helper attribue for the R, Z location of the centroid of the Coil
    width : float, optional
        The width of the magnet. Defaults to 0.01 m.
    height : float, optional
        The height of the magnet. Defaults to 0.01 m.
    mu_hat : float, optional
        The angle of the magnetic moment of the magnet in degrees from the z
        axis. Defaults to 0 degrees clockwise from Z axis (i.e. north pole
        points in the +z direction).
    """

    _SET_GREENS_TRIGS = CurrentFilamentSet._SET_GREENS_TRIGS
    _SET_GREENS_TRIGS += ['r0', 'z0', 'width', 'height', 'mu_hat']

    _codes = [Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CLOSEPOLY]

    def __init__(self, r0, z0, width=0.01, height=0.01, mu_hat=0., **kwargs):
        self.centroid = r0, z0
        self.width = width
        self.height = height
        self.mu_hat = mu_hat
        if kwargs.get('weights', None) is None:
            weights = np.ones(16)
            weights[8:] = -1.
            kwargs['weights'] = weights
        super().__init__(**kwargs)

    @property
    def r0(self):
        return self._r0

    @property
    def z0(self):
        return self._z0

    @property
    def centroid(self):
        return np.array([self.r0, self.z0])

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def mu_hat(self):
        return self._mu_hat

    @property
    def npts(self):
        return len(self._weights)

    @property
    def rz_pts(self):
        # Compute the rz_pts locations from this coil's internal parameters
        npts = self.npts
        hnpts = npts//2
        r0, z0 = self.centroid
        hw, hh = self.width/2, self.height/2
        rspace, zspace = hw*np.ones(hnpts), np.linspace(-hh, hh, hnpts)

        rz_pts = np.empty((npts, 2))
        rz_pts[0:hnpts, 0] = -rspace
        rz_pts[hnpts:, 0] = -rspace
        rz_pts[0:hnpts, 1] = zspace
        rz_pts[hnpts:, 1] = zspace

        if np.isclose(self.mu_hat, 0):
            return rz_pts
        return rotate(rz_pts, self.mu_hat, pivot=(r0, z0))

    @property
    def _verts(self):
        # Get indices for 4 corners of current filament array
        npts = self.npts
        idx = np.array([0, npts//2 - 1, npts - 1, npts//2, 0])
        return self.rz_pts[idx, :]

    @property
    def patch(self):
        return patches.PathPatch(Path(self._verts, self._codes))

    @r0.setter
    def r0(self, r0):
        self._r0 = r0

    @z0.setter
    def z0(self, z0):
        self._z0 = z0

    @centroid.setter
    def centroid(self, centroid):
        self.r0 = centroid[0]
        self.z0 = centroid[1]

    @width.setter
    def width(self, width):
        self._width = width

    @height.setter
    def height(self, height):
        self._height = height

    @mu_hat.setter
    def mu_hat(self, mu_hat):
        self._mu_hat = mu_hat

    def translate(self, vector):
        self.centroid += np.array(vector)

    def rotate(self, angle, pivot=(0., 0.)):
        self.mu_hat += angle
        angle = math.pi*angle / 180
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        self.centroid = (self.centroid - pivot) @ rotation + np.asarray(pivot)
