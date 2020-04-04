from abc import ABCMeta, abstractproperty
import math
import numpy as np


class Grid(metaclass=ABCMeta):
    """A grid in the R-Z plane for calculating magnetic fields and flux.

    The Grid class is a base class for concrete representations of grids that
    may be used in mesh based calculations. These grids can be 1D or 2D
    structured grids as well as a list of arbitrary coordinate pairs as in the
    case of unstructured grids. This class outlines the interface and protocols
    that define a grid in the Pleiades package.

    Attributes
    ----------
    R : np.ndarray
        An N-dimensional array (typically 1 or 2) whose values represent the
        radial coordinates for the mesh in a cylindrical coordinate frame.
    Z : np.ndarray
        An N-dimensional array (typically 1 or 2) whose values represent the
        z coordinates for the mesh in a cylindrical coordinate frame.
    r : np.ndarray
        An N-dimensional array (typically 1 or 2) whose values represent the
        r coordinates for the mesh in a spherical coordinate frame.
    theta : np.ndarray
        An N-dimensional array (typically 1 or 2) whose values represent the
        theta coordinates for the mesh in a spherical coordinate frame.
    """
    def __init__(self):
        pass

    @property
    def R(self):
        return self._R

    @property
    def Z(self):
        return self._Z

    @property
    def r(self):
        return np.sqrt(self._R**2 + self._Z**2)

    @property
    def theta(self):
        return np.cos(self._Z/self.r)

    @property
    def shape(self):
        return self._R.shape


class PointsGrid(Grid):
    """An unstructured grid specified by two lists of coordinate points.

    Parameters
    ----------
    rpts : iterable of float
        List of cylindrical r values for the unstructured grid in meters.
    zpts : iterable of float
        List of z values for the unstructured grid in meters.
    """
    def __init__(self, rpts, zpts):
        assert len(rpts) == len(zpts)
        self._R = np.asarray(rpts).squeeze()
        self._Z = np.asarray(zpts).squeeze()


class RChord(PointsGrid):
    """A 1D chord of cylindrical radii at fixed height z.

    Parameters
    ----------
    rpts : iterable of float
        List of cylindrical r values for the chord in meters.
    z0 : float, optional
        Height of the cylindrical chord in meters. Defaults to 0 m.
    """
    def __init__(self, rpts, z0=0.):
        zpts = z0*np.ones(len(rpts))
        super().__init__(rpts, zpts)


class ZChord(PointsGrid):
    """A 1D chord of Z values at fixed cylindrical radius.

    Parameters
    ----------
    zpts : iterable of float
        List of z values in the chord in meters.
    r0 : float, optional
        Cylindrical radius for the z chord. Defaults to 0 m.
    """
    def __init__(self, zpts, r0=0.):
        rpts = r0*np.ones(len(zpts))
        super().__init__(rpts, zpts)


class SphericalRChord(PointsGrid):
    """A 1D chord of spherical radius at fixed polar angle theta.

    Parameters
    ----------
    rpts : iterable of float
        List of theta points in the chord in degrees.
    theta0 : float, optional
        Polar angle for the radial chord in degrees. Defaults to 0.
    """
    def __init__(self, rpts, theta0=0.):
        rpts = np.asarray(rpts)
        tpts = theta0*np.ones(len(rpts))*math.pi/180
        zpts = np.cos(tpts)*rpts
        rpts = np.sin(tpts)*rpts
        super().__init__(rpts, zpts)


class ThetaChord(PointsGrid):
    """A 1D chord of polar angles at fixed spherical radius r.

    Parameters
    ----------
    tpts : iterable of float
        List of theta points in the chord in degrees.
    r0 : float, optional
        Spherical radius for the chord. Defaults to 1.0 m
    """

    def __init__(self, tpts, r0=1.):
        tpts *= math.pi/180
        tpts = np.asarray(tpts)
        rpts = r0*np.sin(tpts)
        zpts = r0*np.cos(tpts)
        super().__init__(rpts, zpts)


class RectGrid(Grid):
    """A regular linearly spaced 2D R-Z grid.

    Parameters
    ----------
    rmin : float, optional
        The minimum cylindrical radius for the mesh in meters. Defaults to 0 m.
    rmax : float, optional
        The maximum cylindrical radius for the mesh in meters. Defaults to 0 m.
    nr : float, optional
        The number of radial mesh points. Defaults to 101.
    zmin : float, optional
        The minimum z height for the mesh in meters. Defaults to 0 m.
    zmax : float, optional
        The maximum z height for the mesh in meters. Defaults to 0 m.
    nz : float, optional
        The number of z mesh points. Defaults to 101.

    Attributes
    ----------
    rmin : float
        The minimum cylindrical radius for the mesh in meters.
    rmax : float
        The maximum cylindrical radius for the mesh in meters.
    nr : float
        The number of radial mesh points.
    zmin : float
        The minimum z height for the mesh in meters.
    zmax : float
        The maximum z height for the mesh in meters.
    nz : float
        The number of z mesh points.
    """
    def __init__(self, rmin=0., rmax=1., nr=101, zmin=-0.5, zmax=0.5, nz=101):
        self._rmin = rmin
        self._rmax = rmax
        self._nr = nr
        self._zmin = zmin
        self._zmax = zmax
        self._nz = nz
        self._set_grid()

    @property
    def rmin(self):
        return self._rmin

    @property
    def rmax(self):
        return self._rmax

    @property
    def nr(self):
        return self._nr

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax

    @property
    def nz(self):
        return self._nz

    @rmin.setter
    def rmin(self, rmin):
        self._rmin = rmin
        self._set_grid()

    @rmax.setter
    def rmax(self, rmax):
        self._rmax = rmax
        self._set_grid()

    @nr.setter
    def nr(self, nr):
        self._nr = nr
        self._set_grid()

    @zmin.setter
    def zmin(self, zmin):
        self._zmin = zmin
        self._set_grid()

    @zmax.setter
    def zmax(self, zmax):
        self._zmax = zmax
        self._set_grid()

    @nz.setter
    def nz(self, nz):
        self._nz = nz
        self._set_grid()

    def _set_grid(self):
        r = np.linspace(self.rmin, self.rmax, self.nr)
        z = np.linspace(self.zmin, self.zmax, self.nz)
        self._R, self._Z = np.meshgrid(r, z)


#def get_chebnodes(nx,Lx):
#    n = np.ceil((np.sqrt(8*nx+1) - 1)/2.0)
#    if np.mod(np.ceil(n/2),2) == 0.0:
#        if np.mod(n,2) == 0.0:
#            n+=1
#        else:
#            n+=2
#    else:
#        pass
#    n = int(n)
#    zlist = []
#    for i in range(1,n+1):
#        zlist.extend([z for z in us_roots(i)[0]])
#    return Lx*np.array(sorted(zlist))
#
#
#def stretched_grid(nx,Lx,kfac):
#    x = np.zeros(nx)
#    dx = np.zeros(nx)
#    dx1 = (kfac-1)/(kfac**(nx-1)-1)*Lx
#    dx[1] = dx1
#    x[1] = dx1
#    for i in range(2,nx):
#        dx[i] = kfac*dx[i-1]
#        x[i] = x[i-1] + dx[i]
#    return np.abs(x-Lx)[::-1]
