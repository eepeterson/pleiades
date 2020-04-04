from abc import ABCMeta
import numpy as np

from matplotlib.collections import PatchCollection
from pleiades.current_sets import CurrentFilamentSet
from pleiades.fieldmath import compute_greens, compute_greens_2d
from pleiades.mixin import FieldsOperator2D


class Device(FieldsOperator2D):
    """A container for a full configuration of magnets for an experiment

    Parameters
    ----------

    Attributes
    ----------
    mesh : pleiades.Mesh object
        A mesh on which to compute Green's functions and fields
    R : np.array
        The R locations of the mesh
    Z : np.array
        The Z locations of the mesh
    psi : np.array
        The psi values on the mesh
    BR : np.array
        The BR values on the mesh
    BZ : np.array
        The BZ values on the mesh
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

    @property
    def rzw(self):
        return [cset.rzw for cset in self.current_sets]

    @property
    def mesh(self):
        return self._mesh

    @property
    def R(self):
        return self._mesh.R

    @property
    def Z(self):
        return self._mesh.Z

    @property
    def patches(self):
        return [obj.patch for name, obj in self.current_sets]

    @property
    def patch_coll(self):
        return PatchCollection(self.patches, match_original=True)

    @property
    def _uptodate(self):
        return all([obj._uptodate for name, obj in self.current_sets])

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh
        for cset in [obj for name, obj in self.current_sets]:
            cset.mesh = mesh

    def plot_currents(self, ax, **kwargs):
        for cset in [obj for name, obj in self.current_sets]:
            cset.plot(ax, **kwargs)

    def plot_psi(self, ax, *args, **kwargs):
        R, Z = self.mesh.R, self.mesh.Z
        return ax.contour(R, Z, self.psi().reshape(R.shape), *args, **kwargs)

    def plot_modB(self, ax, *args, **kwargs):
        modB = np.sqrt(self.BR()**2 + self.BZ()**2)
        return ax.contour(self.mesh.R, self.mesh.Z, modB, *args, **kwargs)

    def plot(self, ax, *args, **kwargs):
        ax.add_collection(self.patch_coll)
        return self.plot_psi(ax, *args, **kwargs)
