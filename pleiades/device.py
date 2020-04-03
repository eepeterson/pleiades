from abc import ABCMeta
import numpy as np
from matplotlib.collections import PatchCollection
from pleiades import CurrentFilamentSet


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
        m = self.grid.size
        n = len(self.current_sets)
        gpsi = np.empty((m, n))
        gbr = np.empty((m, n))
        gbz = np.empty((m, n))
        for i, cs in enumerate([obj for name, obj in self.current_sets]):
            gpsi[i, :] = cs.gpsi.ravel()
            gbr[i, :] = cs.gBR.ravel()
            gbz[i, :] = cs.gBZ.ravel()

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

    @property
    def psi(self):
        return self.currents @ self.gpsi

    @property
    def BR(self):
        return self.currents @ self.gBR

    @property
    def BZ(self):
        return self.currents @ self.gBZ

    @property
    def gpsi(self):
        if not self._uptodate:
            self._compute_greens()
        return self._gpsi

    @property
    def gBR(self):
        if not self._uptodate:
            self._compute_greens()
        return self._gBR

    @property
    def gBZ(self):
        if not self._uptodate:
            self._compute_greens()
        return self._gBZ

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

    def plot_currents(self, ax):
        for cset in [obj for name, obj in self.current_sets]:
            cset.plot_currents(ax)

    def plot_psi(self, ax, **kwargs):
        R, Z = self.grid.R, self.grid.Z
        return ax.contour(R, Z, self.psi.reshape(R.shape), **kwargs)

    def plot_modB(self, ax, **kwargs):
        modB = np.sqrt(self.BR**2 + self.BZ**2)
        return ax.contour(self.grid.R, self.grid.Z, modB, **kwargs)

    def plot(self, ax, **kwargs):
        ax.add_collection(self.patch_coll)
        return self.plot_psi(ax, **kwargs)
