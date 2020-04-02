from abc import ABCMeta, abstractmethod
from warnings import warn, simplefilter
import math
import numpy as np
from scipy.special import ellipk, ellipe
from matplotlib.path import Path
import matplotlib.patches as patches

from filaments import CurrentFilamentSet


_OUT_OF_DATE_GREENS = """Warning: The greens function for this instance is now
out of date"""

def rotate(pts, angle, pivot=(0., 0.)):
    pivot = np.asarray(pivot)
    angle = math.pi*angle/180
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return (np.asarray(pts) - pivot) @ rotation + pivot


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
        A 4x2 np.array representing the 4 vertices of the coil.
    area : float
        The area of the coil in m^2.
    current_density : float
        The current density in the coil. This is equal to the total current
        divided by the area.
    """
    _codes = [Path.MOVETO,
              Path.LINETO,
              Path.LINETO,
              Path.LINETO,
              Path.CLOSEPOLY]

    def __init__(self, r0, z0, nr=1, nz=1, dr=0.1, dz=0.1, angle=0., **kwargs):
        self.centroid = (r0, z0)
        self.nr = nr
        self.nz = nz
        self.dr = dr
        self.dz = dz
        self.angle = angle
        print(kwargs)
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
    def npts(self):
        return self.nr*self.nz

    @property
    def angle(self):
        return self._angle

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
    def verts(self):
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

    @property
    def patch(self):
        return patches.PathPatch(Path(self.verts, self._codes))

    def translate(self, vector):
        self.centroid += np.array(vector)

    def rotate(self, angle, pivot=(0., 0.)):
        self.angle += angle
        angle = math.pi*angle / 180
        c, s = np.cos(angle), np.sin(angle)
        rotation = np.array([[c, -s], [s, c]])
        self.centroid = (self.centroid - pivot) @ rotation + np.asarray(pivot)


#class MagnetRing(CurrentFilamentSet):
#    """Represent a Rectangular cross-section dipole magnet with axisymmetric 
#    surface currents.
#
#    Parameters
#    ----------
#    loc : tuple
#        The (R, Z) location of the centroid of the magnet.
#    width : float, optional
#        The width of the magnet. Defaults to 0.01 m.
#    height : float, optional
#        The height of the magnet. Defaults to 0.01 m.
#    mu_hat : float, optional
#        The angle of the magnetic moment of the magnet in degrees from the z
#        axis. Defaults to 0 degrees clockwise from Z axis (i.e. north pole
#        points in the +z direction).
#
#    Attributes
#    ----------
#    loc : tuple
#        The (R, Z) location of the centroid of the magnet.
#    width : float, optional
#        The width of the magnet. Defaults to 0.01 m.
#    height : float, optional
#        The height of the magnet. Defaults to 0.01 m.
#    mu_hat : float, optional
#        The angle of the magnetic moment of the magnet in degrees from the z
#        axis. Defaults to 0 degrees clockwise from Z axis (i.e. north pole
#        points in the +z direction).
#    current_prof : integer or array_like
#        The current profile along the side of the magnet. Defaults to
#        np.ones(8) i.e. 8 equal surface currents per side.
#    """
#
#    def __init__(self, loc, width=0.01, height=0.01, mu=1., mu_hat=0., **kwargs):
#        self.loc = loc
#        self.width = width
#        self.height = height
#        self.mu = mu
#        self.mu_hat = mu_hat
#        super().__init__(rz_pts, **kwargs)
#
#    def reset(self, **kwargs):
#        # set Magnet specific attributes before calling super constructor
#        r0, z0 = kwargs.pop("loc", (1.0, 1.0))
#        if r0 < 0:
#            raise ValueError("Centroid of magnet, r0, must be >= 0")
#        r0 = float(r0)
#        z0 = float(z0)
#        width = float(kwargs.pop("width", .01))
#        height = float(kwargs.pop("height", .01))
#        if not (width > 0 and height > 0):
#            raise ValueError("width and height must be greater than 0")
#        self._width = width
#        self._height = height
#        ## need to pop this now but save it for later
#        mu_hat = kwargs.pop("mu_hat", 0)
#        self._mu_hat = 0
#        current_prof = kwargs.pop("current_prof", 10)
#        if isinstance(current_prof, Number):
#            current_prof = ones(current_prof)
#        else:
#            current_prof = array(current_prof)
#        if not current_prof.size > 0:
#            raise ValueError("current_prof array must have size > 0")
#        self._current_prof = current_prof
#        self._loc = (r0, z0)
#        # start building super class relevant inputs
#        # super_kwargs include rz_pts,current,patchcls,patchargs_dict, any matplotlib.patches kwarg
#        current = kwargs.pop("current", 1)
#        if not current > 0:
#            raise ValueError("current must be > 0")
#        self._current = current
#        n = len(self._current_prof)
#        dummy = ones(n)
#        rpts = self._width / 2.0 * hstack((-1 * dummy, dummy))
#        if n == 1:
#            zpts = zeros(2)
#        else:
#            ztmp = linspace(-self._height / 2.0, self._height / 2.0, n)
#            zpts = hstack((ztmp, ztmp))
#        rz_pts = vstack((rpts + r0, zpts + z0)).T
#        patchkwargs = {"closed": True, "fc": "w", "ec": "k", "zorder": 3}
#        # All leftover kwargs get put into patchkwargs
#        patchkwargs.update(kwargs)
#        # Build kwargs for super constructor
#        super_kwargs = {"rz_pts": rz_pts, "current": 1.0, "patchcls": Polygon, "patchargs_dict": {}}
#        super_kwargs.update(patchkwargs)
#        # builds CurrentGroup at loc with current = 1 for all current objs
#        super(Magnet, self).__init__(**super_kwargs)
#        # make left side currents negative (current setter overridden below)
#        self.current = self._current
#        # rotate according to muhat direction
#        self.mu_hat = mu_hat
#
#    @property
#    def loc(self):
#        return self._loc
#
#    @property
#    def width(self):
#        return self._width
#
#    @property
#    def height(self):
#        return self._height
#
#    @property
#    def mu_hat(self):
#        return self._mu_hat
#
#    @property
#    def current_prof(self):
#        return self._current_prof
#
#    @loc.setter
#    def loc(self, r0, z0):
#        self._loc = (r0, z0)
#
#    @width.setter
#    def width(self, width):
#        self._width = width
#
#    @height.setter
#    def height(self, height):
#        self._height = height
#
#    @mu_hat.setter
#    def mu_hat(self, mu_hat):
#        self.rotate(mu_hat - self._mu_hat)
#
#    @current_prof.setter
#    def current_prof(self, new_prof):
#        self.rebuild("current_prof", new_prof)


class CurrentFilamentMultiSet(metaclass=ABCMeta):
    """A helper class for storing multiple CurrentFilamentSets in one object
    each with their own current value and Green's function.
    """
    pass


