from abc import ABCMeta, abstractproperty
from collections.abc import Iterable
import math
import numpy as np


class Mesh(metaclass=ABCMeta):
    """A mesh in the R-Z plane for calculating magnetic fields and flux.

    The Mesh class is a base class for concrete representations of meshs that
    may be used in mesh based calculations. These meshs can be 1D or 2D
    structured meshs as well as a list of arbitrary coordinate pairs as in the
    case of unstructured meshs. This class outlines the interface and protocols
    that define a mesh in the Pleiades package.

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
        self._R = None
        self._Z = None

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

    @classmethod
    def to_points(cls, mesh):
        """Take a mesh or numpy array, return Nx2 points array"""
        if isinstance(mesh, cls):
            shape = mesh.R.shape
            rz_pts = np.empty((mesh.R.size, 2))
            rz_pts[:, 0] = mesh.R.ravel()
            rz_pts[:, 1] = mesh.Z.ravel()
        elif isinstance(mesh, np.ndarray):
            assert len(mesh.shape) == 2
            assert mesh.shape[1] == 2
            shape = mesh.shape[0:1]
            rz_pts = mesh
        elif isinstance(mesh, Iterable):
            assert len(mesh) == 2
            shape = mesh[0].shape
            rz_pts = np.empty_like(mesh[0])
            rz_pts[:, 0] = mesh[0].ravel()
            rz_pts[:, 1] = mesh[1].ravel()
        else:
            raise ValueError('Unsupported type for mesh')

        return rz_pts

    @classmethod
    def from_array(cls, mesh):
        mesh = np.asarray(mesh)
        if mesh.dim == 2:
            if mesh.shape[1] == 2:
                rpts, zpts = mesh[:, 0], mesh[1, :]
            else:
                rpts, zpts = mesh[0, :], mesh[:, 1]
            shape = rpts.shape
            return PointsMesh(rpts, zpts)
        elif mesh.dim == 3:
            if mesh.shape[0] == 2:
                rpts, zpts = mesh[0, :, :], mesh[1, :, :]
                shape = rpts.shape
                kwargs = {'rmin': np.amin(rpts), 'rmax': np.amax(rpts),
                          'zmin': np.amin(zpts), 'zmax': np.amax(zpts),
                          'nr': rpts.shape[1], 'nz': rpts.shape[0]}

                return RectMesh(**kwargs)
            else:
                raise ValueError


class PointsMesh(Mesh):
    """An unstructured mesh specified by two lists of coordinate points.

    Parameters
    ----------
    rpts : iterable of float
        List of cylindrical r values for the unstructured mesh in meters.
    zpts : iterable of float
        List of z values for the unstructured mesh in meters.
    """
    def __init__(self, rpts, zpts):
        assert len(rpts) == len(zpts)
        self._R = np.asarray(rpts).squeeze()
        self._Z = np.asarray(zpts).squeeze()


class RChord(PointsMesh):
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


class ZChord(PointsMesh):
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


class SphericalRChord(PointsMesh):
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


class ThetaChord(PointsMesh):
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


class RectMesh(Mesh):
    """A regular linearly spaced 2D R-Z mesh.

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
        super().__init__()
        self._rmin = rmin
        self._rmax = rmax
        self._nr = nr
        self._zmin = zmin
        self._zmax = zmax
        self._nz = nz
        self._set_mesh()

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
        self._set_mesh()

    @rmax.setter
    def rmax(self, rmax):
        self._rmax = rmax
        self._set_mesh()

    @nr.setter
    def nr(self, nr):
        self._nr = nr
        self._set_mesh()

    @zmin.setter
    def zmin(self, zmin):
        self._zmin = zmin
        self._set_mesh()

    @zmax.setter
    def zmax(self, zmax):
        self._zmax = zmax
        self._set_mesh()

    @nz.setter
    def nz(self, nz):
        self._nz = nz
        self._set_mesh()

    def _set_mesh(self):
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
#def stretched_mesh(nx,Lx,kfac):
#    x = np.zeros(nx)
#    dx = np.zeros(nx)
#    dx1 = (kfac-1)/(kfac**(nx-1)-1)*Lx
#    dx[1] = dx1
#    x[1] = dx1
#    for i in range(2,nx):
#        dx[i] = kfac*dx[i-1]
#        x[i] = x[i-1] + dx[i]
#    return np.abs(x-Lx)[::-1]
