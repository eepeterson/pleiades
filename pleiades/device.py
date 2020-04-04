from abc import ABCMeta
import numpy as np
from matplotlib.collections import PatchCollection
from pleiades import CurrentFilamentSet, compute_greens, compute_greens_2d


class Device(metaclass=ABCMeta):
    """A container for a full configuration of magnets for an experiment

    Parameters
    ----------

    Attributes
    ----------
    grid : pleiades.Grid object
        A grid on which to compute Green's functions and fields
    R : np.array
        The R locations of the grid
    Z : np.array
        The Z locations of the grid
    psi : np.array
        The psi values on the grid
    BR : np.array
        The BR values on the grid
    BZ : np.array
        The BZ values on the grid
    patches : list
        A list of patch objects for the configuration
    patch_coll : matplotlib.patches.PatchCollection
        A patch collection for easier adding to matplotlib axes
    """

    def __init__(self, **kwargs):
        self._current_sets = []

    def __setattr__(self, name, value):
        if isinstance(value, CurrentFilamentSet):
            if name not in self._current_sets:
                self._current_sets.append([name, value])
            else:
                raise ValueError
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._current_sets:
            self._current_sets.remove([name, getattr(self, name)])
        super().__delattr__(name)

    @property
    def current_sets(self):
        return self._current_sets

    @property
    def currents(self):
        return np.array([obj.current for name, obj in self.current_sets])

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

    @property
    def grid(self):
        return self._grid

    @property
    def R(self):
        return self._grid.R

    @property
    def Z(self):
        return self._grid.Z

    def gpsi(self, rz_pts=None):
        """Compute the Green's function for magnetic flux, :math:`psi`.

        Parameters
        ----------
        rz_pts : ndarray, optional
            An Nx2 array of points representing (R, Z) coordinates at which to
            calculate the magnetic flux. Defaults to None, in which case the
            CurrentFilamentSet.grid attribute is used.

        Returns
        -------
        gpsi : ndarray
            1D array representing the Green's function for flux and whose size
            is equal to the number of rz_pts.
        """
        if rz_pts is None:
            if not self._uptodate:
                self._compute_greens()
            return self._gpsi
        return compute_greens_2d([c.rzw for c in self.current_sets], rz_pts)[0]

    def gBR(self, rz_pts=None):
        """Compute the Green's function for the radial magnetic field, BR

        Parameters
        ----------
        rz_pts : ndarray, optional
            An Nx2 array of points representing (R, Z) coordinates at which to
            calculate BR. Defaults to None, in which case the
            CurrentFilamentSet.grid attribute is used.

        Returns
        -------
        gBR : ndarray
            1D array representing the Green's function for BR and whose size
            is equal to the number of rz_pts.
        """
        if rz_pts is None:
            if not self._uptodate:
                self._compute_greens()
            return self._gBR
        return compute_greens_2d([c.rzw for c in self.current_sets], rz_pts)[1]

    def gBZ(self, rz_pts=None):
        """Compute the Green's function for the vertical magnetic field, BZ

        Parameters
        ----------
        rz_pts : ndarray, optional
            An Nx2 array of points representing (R, Z) coordinates at which to
            calculate BZ. Defaults to None, in which case the
            CurrentFilamentSet.grid attribute is used.

        Returns
        -------
        gBZ : ndarray
            1D array representing the Green's function for BZ and whose size
            is equal to the number of rz_pts.
        """
        if rz_pts is None:
            if not self._uptodate:
                self._compute_greens()
            return self._gBZ
        return compute_greens_2d([c.rzw for c in self.current_sets], rz_pts)[2]

    def psi(self, currents=None, rz_pts=None):
        """Compute the magnetic flux, :math:`psi`.

        Parameters
        ----------
        current : float, optional
            Specify a current value in amps to use instead of
            CurrentFilamentSet.current. Defaults to None, in which case the
            current attribute is used to calculate the flux.
        rz_pts : ndarray, optional
            An Nx2 array of points representing (R, Z) coordinates at which to
            calculate the magnetic flux. Defaults to None, in which case the
            CurrentFilamentSet.grid attribute is used.

        Returns
        -------
        psi : ndarray
        """
        currents = self.currents if currents is None else currents
        return currents @ self.gpsi(rz_pts=rz_pts)

    def BR(self, currents=None, rz_pts=None):
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
        return currents @ self.gBR(rz_pts=rz_pts)

    def BZ(self, currents=None, rz_pts=None):
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
        return currents @ self.gBZ(rz_pts=rz_pts)

    @property
    def patches(self):
        return [obj.patch for name, obj in self.current_sets]

    @property
    def patch_coll(self):
        return PatchCollection(self.patches, match_original=True)

    @property
    def _uptodate(self):
        return all([obj._uptodate for name, obj in self.current_sets])

    @grid.setter
    def grid(self, grid):
        self._grid = grid
        for cset in [obj for name, obj in self.current_sets]:
            cset.grid = grid

    def plot_currents(self, ax, **kwargs):
        for cset in [obj for name, obj in self.current_sets]:
            cset.plot(ax, **kwargs)

    def plot_psi(self, ax, *args, **kwargs):
        R, Z = self.grid.R, self.grid.Z
        return ax.contour(R, Z, self.psi().reshape(R.shape), *args, **kwargs)

    def plot_modB(self, ax, *args, **kwargs):
        modB = np.sqrt(self.BR()**2 + self.BZ()**2)
        return ax.contour(self.grid.R, self.grid.Z, modB, *args, **kwargs)

    def plot(self, ax, *args, **kwargs):
        ax.add_collection(self.patch_coll)
        return self.plot_psi(ax, *args, **kwargs)
