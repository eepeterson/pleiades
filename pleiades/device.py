from abc import ABCMeta
import numpy as np

from matplotlib.collections import PatchCollection
from pleiades.current_sets import CurrentFilamentSet
from pleiades.fields import FieldsOperator


class Device(FieldsOperator):
    """A container for a full configuration of magnets for an experiment

    Parameters
    ----------
    **kwargs :
        Any valid argument for FieldsOperator

    Attributes
    ----------
    current_sets: iterable
        List of CurrentFilamentSet objects
    R : np.array
        The R locations of the mesh
    Z : np.array
        The Z locations of the mesh
    patches : list
        A list of patch objects for the configuration
    patch_coll : matplotlib.patches.PatchCollection
        A patch collection for easier adding to matplotlib axes
    """

    def __init__(self, **kwargs):
        self._current_sets = []
        super().__init__(rank=2, **kwargs)

    def __setattr__(self, name, value):
        if isinstance(value, CurrentFilamentSet):
            if value not in self:
                self._current_sets.append(name)
            else:
                raise AttributeError('this attribute is already set')
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._current_sets:
            self._current_sets.remove(name)
        super().__delattr__(name)

    def __iter__(self):
        for cset in self._current_sets:
            yield getattr(self, cset)

    @property
    def current_sets(self):
        return list(self)

    @property
    def current(self):
        return np.array([cset.current for cset in self])

    @property
    def rzw(self):
        return [cset.rzw for cset in self]

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
        return [cset.patch for cset in self]

    @property
    def patch_coll(self):
        return PatchCollection(self.patches, match_original=True)

    @property
    def _uptodate(self):
        return all([cset._uptodate for cset in self])

    @mesh.setter
    def mesh(self, mesh):
        self._mesh = mesh
        for cset in self:
            cset.mesh = mesh

    def plot_currents(self, ax, **kwargs):
        for cset in self:
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
