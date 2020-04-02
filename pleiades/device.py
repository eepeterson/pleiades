from abc import ABCMeta
import numpy as np
from matplotlib.collections import PatchCollection


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
        self._filament_sets = []

    def __setattr__(self, name, value):
        if isinstance(value, CurrentFilamentSet):
            if name not in self._filament_sets:
                self._filament_sets.append(name)
            else:
                raise ValueError
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._filament_sets:
            self._filament_sets.remove(name)
        super().__delattr__(name)

    @property
    def filament_sets(self):
        return self._filament_sets

    @property
    def currents(self):
        return np.array([getattr(self, f).current for f in self.filament_sets])

    def compute_greens(self, R, Z):
        """Helper function for computing Green's functions

        Parameters
        ----------
        R : np.array
            A 1D np.array representing the R positions of the grid
        Z : np.array
            A 1D np.array representing the Z positions of the grid
        """
        m = len(R.ravel())
        n = len(self.filament_sets)
        gpsi = np.empty((m, n))
        gbr = np.empty((m, n))
        gbz = np.empty((m, n))
        for i, fs in enumerate(self.filament_sets):
            gtup = getattr(self, fs).compute_greens(R, Z, return_greens=True)
            gpsi[:, i] = gtup[0]
            gbr[:, i] = gtup[1]
            gbz[:, i] = gtup[2]

        self.g_psi = gpsi
        self.g_br = gbr
        self.g_bz = gbz

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
        return self.g_psi @ self.currents

    @property
    def BR(self):
        return self.g_br @ self.currents

    @property
    def BZ(self):
        return self.g_bz @ self.currents

    @property
    def patches(self):
        return [getattr(self, f).patch for f in self.filament_sets]

    @property
    def patch_coll(self):
        return PatchCollection(self.patches, match_original=True)

    @grid.setter
    def grid(self, grid):
        self._grid = grid
        self.compute_greens(self.grid.R, self.grid.Z)
#        for fs in [getattr(self, f) for f in self.filament_sets]:
#            fs.compute_greens(self.R, self.Z)

    def plot_currents(self, ax):
        for fs in [getattr(self, f) for f in self.filament_sets]:
            fs.plot_currents(ax)

    def plot_psi(self, ax, **kwargs):
        R, Z = self.grid.R, self.grid.Z
        return ax.contour(R, Z, self.psi.reshape(R.shape), **kwargs)

    def plot_modB(self, ax, **kwargs):
        modB = np.sqrt(self.BR**2 + self.BZ**2)
        return ax.contour(self.grid.R, self.grid.Z, modB, **kwargs)

    def plot(self, ax, **kwargs):
        ax.add_collection(self.patch_coll)
        return self.plot_psi(ax, **kwargs)
