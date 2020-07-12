from abc import ABCMeta, abstractmethod, abstractproperty
from collections import Iterable
from warnings import warn, simplefilter
import math
import numpy as np
from scipy.special import ellipk, ellipe
from multiprocessing import Pool, sharedctypes

from pleiades.mesh import Mesh
import pleiades.checkvalue as cv
from pleiades.transforms import rotate


class FieldsOperator(metaclass=ABCMeta):
    """Mixin class for computing fields on meshes

    Parameters
    ----------
    mesh : pleiades.Mesh object, optional
        The mesh to use for calculating fields
    rank : int (1 or 2)
        Indicator of whether the current attribute is a scalar or vector

    Variables
    ---------
    current : float or ndarray
        Current values in this object
    rzw : ndarray or list of ndarray
        Nx3 arrays of centroid positions and weights
    mesh : pleiades.Mesh object
        The mesh to use for calculating fields
    """

    def __init__(self, mesh=None, rank=1, **kwargs):
        # mesh should accept 2d, 3d or 2 1d or 2 2d)
        self._gpsi = None
        self._gBR = None
        self._gBZ = None
        if rank == 1:
            self._uptodate = False
        self.rank = rank
        self.mesh = mesh

    @abstractproperty
    def current(self):
        pass

    @abstractproperty
    def rzw(self):
        pass

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    @cv.flag_greens_on_set
    def mesh(self, mesh):
        if not isinstance(mesh, Mesh) and mesh is not None:
            mesh = Mesh.from_array(mesh)
        self._mesh = mesh

    def gpsi(self, mesh=None):
        """Compute the Green's function for magnetic flux, :math:`psi`.

        Parameters
        ----------
        mesh : ndarray, optional
            An Nx2 array of points representing (R, Z) coordinates at which to
            calculate the magnetic flux. Defaults to None, in which case the
            CurrentFilamentSet.mesh attribute is used.

        Returns
        -------
        gpsi : ndarray
            1D array representing the Green's function for flux and whose size
            is equal to the number of mesh.
        """
        if mesh is None:
            if not self._uptodate:
                self._compute_greens()
            return self._gpsi
        return compute_greens(self.rzw, Mesh.to_points(mesh))[0]

    def gBR(self, mesh=None):
        """Compute the Green's function for the radial magnetic field, BR

        Parameters
        ----------
        mesh : ndarray, optional
            An Nx2 array of points representing (R, Z) coordinates at which to
            calculate BR. Defaults to None, in which case the
            CurrentFilamentSet.mesh attribute is used.

        Returns
        -------
        gBR : ndarray
            1D array representing the Green's function for BR and whose size
            is equal to the number of mesh.
        """
        if mesh is None:
            if not self._uptodate:
                self._compute_greens()
            return self._gBR
        return compute_greens(self.rzw, Mesh.to_points(mesh))[1]

    def gBZ(self, mesh=None):
        """Compute the Green's function for the vertical magnetic field, BZ

        Parameters
        ----------
        mesh : ndarray, optional
            An Nx2 array of points representing (R, Z) coordinates at which to
            calculate BZ. Defaults to None, in which case the
            CurrentFilamentSet.mesh attribute is used.

        Returns
        -------
        gBZ : ndarray
            1D array representing the Green's function for BZ and whose size
            is equal to the number of mesh.
        """
        if mesh is None:
            if not self._uptodate:
                self._compute_greens()
            return self._gBZ
        return compute_greens(self.rzw, Mesh.to_points(mesh))[2]

    def psi(self, current=None, mesh=None):
        """Compute the magnetic flux, :math:`psi`.

        Parameters
        ----------
        current : float, optional
            Specify a current value in amps to use instead of
            CurrentFilamentSet.current. Defaults to None, in which case the
            current attribute is used to calculate the flux.
        mesh : ndarray, optional
            An Nx2 array of points representing (R, Z) coordinates at which to
            calculate the magnetic flux. Defaults to None, in which case the
            CurrentFilamentSet.mesh attribute is used.

        Returns
        -------
        psi : ndarray
        """
        current = current if current is not None else self.current
        if self.rank == 1:
            return current*self.gpsi(mesh=mesh)

        if self.rank == 2:
            return current @ self.gpsi(mesh=mesh)

    def BR(self, current=None, mesh=None):
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
        current = current if current is not None else self.current
        if self.rank == 1:
            return current*self.gBR(mesh=mesh)

        if self.rank == 2:
            return current @ self.gBR(mesh=mesh)

    def BZ(self, current=None, mesh=None):
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
        current = current if current is not None else self.current
        if self.rank == 1:
            return current*self.gBZ(mesh=mesh)

        if self.rank == 2:
            return current @ self.gBZ(mesh=mesh)

    def _compute_greens(self):
        """Compute and assign the Green's functions for psi, BR, and BZ"""
        # Calculate Green's functions 
        if self.rank == 1:
            gpsi, gBR, gBZ = compute_greens(self.rzw, Mesh.to_points(self.mesh))

        if self.rank == 2:
            m = len(self.current)
            n = len(self.R.ravel())
            gpsi = np.empty((m, n))
            gBR = np.empty((m, n))
            gBZ = np.empty((m, n))
            for i, cset in enumerate(self):
                gpsi[i, :] = cset.gpsi().ravel()
                gBR[i, :] = cset.gBR().ravel()
                gBZ[i, :] = cset.gBZ().ravel()

        self._gpsi = gpsi
        self._gBR = gBR
        self._gBZ = gBZ
        # Notify instance that the Green's functions are up to date only if it's
        # rank 1. Rank 2 FieldOperators get their status from associated rank 1s
        if self.rank == 1:
            self._uptodate = True


def compute_greens(rzw, rz_pts):
    """Compute axisymmetric Green's functions for magnetic fields

    Parameters
    ----------
    rzw: ndarray or iterable of ndarray
        An Nx3 array whose columns are r locations, z locations, and current
        weights respectively for the current filaments.
    rz_pts: Nx2 np.array
        An Nx2 array whose columns are r locations and z locations for the mesh
        points where we want to calculate the Green's functions.

    Returns
    -------
    tuple :
        3-tuple of 1D np.array representing the Green's function for psi, BR,
        and Bz respectively.
    """
    if isinstance(rzw, list):
        return _compute_greens_2d(rzw, rz_pts)
    else:
        return _compute_greens_1d(rzw, rz_pts)


def _compute_greens_1d(rzw, rz_pts):
    """Compute axisymmetric Green's functions for magnetic fields

    Parameters
    ----------
    rzw: Nx3 np.array
        An Nx3 array whose columns are r locations, z locations, and current
        weights respectively for the current filaments.
    rz_pts: Nx2 np.array
        An Nx2 array whose columns are r locations and z locations for the mesh
        points where we want to calculate the Green's functions.

    Returns
    -------
    tuple :
        3-tuple of 1D np.array representing the Green's function for psi, BR,
        and Bz respectively.
    """
    simplefilter('ignore', RuntimeWarning)

    # Begin calculation of Green's functions based on vector potential
    # psi = R*A_phi from a current loop at r0, z0 on a mesh specified by
    # r and z in cylindrical coordinates and with SI units.
    r, z = rz_pts[:, 0], rz_pts[:, 1]
    n = len(r)
    gpsi = np.zeros(n)
    gBR = np.zeros(n)
    gBZ = np.zeros(n)
    r2 = r*r

    # Prefactor c1 for vector potential is mu_0/4pi = 1E-7
    c1 = 1E-7
    for r0, z0, wgt in rzw:
        # Check if the coil position is close to 0 if so skip it
        if np.isclose(r0, 0, rtol=0, atol=1E-12):
            continue

        # Compute factors that are reused in equations
        fac0 = (z - z0)*(z - z0)
        d = np.sqrt(fac0 + (r + r0)*(r + r0))
        d_ = np.sqrt(fac0 + (r - r0)*(r - r0))
        k_2 = 4*r*r0 / (d*d)
        K = ellipk(k_2)
        E = ellipe(k_2)
        denom = d*d_ *d_
        fac1 = K*d_ *d_
        fac2 = (fac0 + r2 + r0*r0)*E

        # Compute Green's functions for psi, BR, BZ
        gpsi_tmp = wgt*c1*r*r0*4 / d / k_2*((2 - k_2)*K - 2*E)
        gBR_tmp = -2*wgt*c1*(z - z0)*(fac1 - fac2) / (r*denom)
        gBZ_tmp = 2*wgt*c1*(fac1 - (fac2 - 2*r0*r0*E)) / denom

        # Correct for infinities and add sum
        gpsi_tmp[~np.isfinite(gpsi_tmp)] = 0
        gpsi += gpsi_tmp
        gBR_tmp[~np.isfinite(gBR_tmp)] = 0
        gBR += gBR_tmp
        gBZ_tmp[~np.isfinite(gBZ_tmp)] = 0
        gBZ += gBZ_tmp

    return gpsi, gBR, gBZ


def _compute_greens_2d(rzw_list, rz_pts):
    """Compute axisymmetric Green's functions for magnetic fields

    Parameters
    ----------
    rzw: list
        A list of Nx3 arrays whose columns are r locations, z locations, and
        current weights respectively for the current filaments.
    rz_pts: Nx2 np.array
        An Nx2 array whose columns are r locations and z locations for the mesh
        points where we want to calculate the Green's functions.

    Returns
    -------
    tuple :
        3-tuple of 1D np.array representing the Green's function for psi, BR,
        and Bz respectively.
    """
    simplefilter('ignore', RuntimeWarning)

    # Begin calculation of Green's functions based on vector potential
    # psi = R*A_phi from a current loop at r0, z0 on a mesh specified by
    # r and z in cylindrical coordinates and with SI units.
    r, z = rz_pts[:, 0], rz_pts[:, 1]
    n = len(r)
    m = len(rzw_list)
    gpsi = np.zeros(m, n)
    gBR = np.zeros(m, n)
    gBZ = np.zeros(m, n)
    r2 = r*r

    # Prefactor c1 for vector potential is mu_0/4pi = 1E-7
    c1 = 1E-7
    for i in range(m):
        for r0, z0, wgt in rzw_list[i]:
            # Check if the coil position is close to 0 if so skip it
            if np.isclose(r0, 0, rtol=0, atol=1E-12):
                continue

            # Compute factors that are reused in equations
            fac0 = (z - z0)*(z - z0)
            d = np.sqrt(fac0 + (r + r0)*(r + r0))
            d_ = np.sqrt(fac0 + (r - r0)*(r - r0))
            k_2 = 4*r*r0 / (d*d)
            K = ellipk(k_2)
            E = ellipe(k_2)
            denom = d*d_ *d_
            fac1 = K*d_ *d_
            fac2 = (fac0 + r2 + r0*r0)*E

            # Compute Green's functions for psi, BR, BZ
            gpsi_tmp = wgt*c1*r*r0*4 / d / k_2*((2 - k_2)*K - 2*E)
            gBR_tmp = -2*wgt*c1*(z - z0)*(fac1 - fac2) / (r*denom)
            gBZ_tmp = 2*wgt*c1*(fac1 - (fac2 - 2*r0*r0*E)) / denom

            # Correct for infinities and add sum
            gpsi_tmp[~np.isfinite(gpsi_tmp)] = 0
            gpsi[i, :] += gpsi_tmp
            gBR_tmp[~np.isfinite(gBR_tmp)] = 0
            gBR[i, :] += gBR_tmp
            gBZ_tmp[~np.isfinite(gBZ_tmp)] = 0
            gBZ[i, :] += gBZ_tmp

    return gpsi, gBR, gBZ


def _compute_greens_mp(rzw, rz_pts):
    # Multiprocessing version
    size = rz_pts.shape[0]
    block_size = 100000
    r, z = rz_pts[:,0], rz_pts[:, 1]
    r2 = r*r

    result = np.ctypeslib.as_ctypes(np.zeros((3, size)))
    shared_array = sharedctypes.RawArray(result._type, result)

    def fill_per_window(window_y):
        tmp = np.ctypeslib.as_array(shared_array)

        simplefilter('ignore', RuntimeWarning)

        # Prefactor c1 for vector potential is mu_0/4pi = 1E-7
        c1 = 1E-7
        for idx_y in range(window_y, window_y + block_size):
            for r0, z0, wgt in rzw:
                # Check if the coil position is close to 0 if so skip it
                if np.isclose(r0, 0, rtol=0, atol=1E-12):
                    continue

                # Compute factors that are reused in equations
                fac0 = (z - z0)*(z - z0)
                d = np.sqrt(fac0 + (r + r0)*(r + r0))
                d_ = np.sqrt(fac0 + (r - r0)*(r - r0))
                k_2 = 4*r*r0 / (d*d)
                K = ellipk(k_2)
                E = ellipe(k_2)
                denom = d*d_ *d_
                fac1 = K*d_ *d_
                fac2 = (fac0 + r2 + r0*r0)*E

                # Compute Green's functions for psi, BR, BZ
                gpsi_tmp = wgt*c1*r*r0*4 / d / k_2*((2 - k_2)*K - 2*E)
                gBR_tmp = -2*wgt*c1*(z - z0)*(fac1 - fac2) / (r*denom)
                gBZ_tmp = 2*wgt*c1*(fac1 - (fac2 - 2*r0*r0*E)) / denom
                gpsi_tmp[~np.isfinite(gpsi_tmp)] = 0
                gBR_tmp[~np.isfinite(gBR_tmp)] = 0
                gBZ_tmp[~np.isfinite(gBZ_tmp)] = 0

                tmp[0, idx_y] += gpsi_tmp
                tmp[1, idx_y] += gBR_tmp
                tmp[2, idx_y] += gBZ_tmp


    window_idxs = [(i, j) for i, j in
                   zip(range(0, size, block_size),
                       range(block_size, size + block_size, block_size))]

    p = Pool()
    res = p.map(fill_per_window, window_idxs)
    result = np.ctypeslib.as_array(shared_array)

    return result[0, :], result[1, :], result[2, :]
