from abc import ABCMeta, abstractmethod, abstractproperty
from collections import Iterable
from warnings import warn, simplefilter
import math
import numpy as np
from scipy.special import ellipk, ellipe
from matplotlib.path import Path
import matplotlib.patches as patches

from pleiades.mesh import Mesh
from pleiades.fieldmath import compute_greens
import pleiades.checkvalue as cv
from pleiades.transforms import rotate


class FieldsOperator(metaclass=ABCMeta):
    """Functionality for Green's functions and mesh handling
    """

    def __init__(self, mesh=None, **kwargs):
        # mesh should accept 2d, 3d or 2 1d or 2 2d)
        self._gpsi = None
        self._gBR = None
        self._gBZ = None
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
    def mesh(self, mesh):
        if not isinstance(mesh, Mesh) and mesh is not None:
            mesh = Mesh.from_array(mesh)
        self._mesh = mesh
        self._uptodate = False

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
        return current*self.gpsi(mesh=mesh)

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
        return current*self.gBR(mesh=mesh)

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
        return current*self.gBZ(mesh=mesh)

    def _compute_greens(self):
        """Compute and assign the Green's functions for psi, BR, and BZ"""
        # Calculate Green's functions 
        gpsi, gBR, gBZ = compute_greens(self.rzw, Mesh.to_points(self.mesh))
        self._gpsi = gpsi
        self._gBR = gBR
        self._gBZ = gBZ

        # Notify instance that the Green's functions are up to date
        self._uptodate = True


class FieldsOperator2D(metaclass=ABCMeta):
    """Functionality for Green's functions and mesh handling
    """

    def __init__(self, mesh=None, **kwargs):
        # mesh should accept 2d, 3d or 2 1d or 2 2d)
        self._gpsi = None
        self._gBR = None
        self._gBZ = None
        self.mesh = mesh

    @abstractproperty
    def currents(self):
        pass

    @abstractproperty
    def rzw(self):
        pass

    @property
    def mesh(self):
        return self._mesh

    @mesh.setter
    def mesh(self, mesh):
        if not isinstance(mesh, Mesh) and mesh is not None:
            mesh = Mesh.from_array(mesh)
        self._mesh = mesh
        self._uptodate = False

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
        return compute_greens_2d(self.rzw, Mesh.to_points(mesh))[0]

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
        return compute_greens_2d(self.rzw, Mesh.to_points(mesh))[1]

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
        return compute_greens_2d(self.rzw, Mesh.to_points(mesh))[2]

    def psi(self, currents=None, mesh=None):
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
        currents = self.currents if currents is None else currents
        return currents @ self.gpsi(mesh=mesh)

    def BR(self, currents=None, mesh=None):
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
        currents = self.currents if currents is None else currents
        return currents @ self.gBR(mesh=mesh)

    def BZ(self, currents=None, mesh=None):
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
        currents = self.currents if currents is None else currents
        return currents @ self.gBZ(mesh=mesh)

    def _compute_greens(self):
        """Compute Green's function matrices for all CurrentFilamentSets"""
        m = len(self.current_sets)
        n = len(self.R.ravel())
        gpsi = np.empty((m, n))
        gbr = np.empty((m, n))
        gbz = np.empty((m, n))
        for i, cs in enumerate([obj for name, obj in self.current_sets]):
            gpsi[i, :] = cs.gpsi().ravel()
            gbr[i, :] = cs.gBR().ravel()
            gbz[i, :] = cs.gBZ().ravel()

        self._gpsi = gpsi
        self._gBR = gbr
        self._gBZ = gbz
