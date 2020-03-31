from abc import ABCMeta


class Device(metaclass=ABCMeta):
    """A container for a full configuration of magnets for an experiment

    Parameters
    ----------
    components : list
        A list of pleiades.Component objects to be added to the configuration
    labels : list of str
        A list of the names for the components being added
    filename : str
        A filename for the Configuration
    grid : pleiades.Grid object
        A grid on which to compute Green's functions and fields
    artists :
        A list of matplotlib patch objects

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
        self.components = kwargs.pop("components", [])
        self.labels = kwargs.pop("labels", [])
        self.filename = kwargs.pop("filename", None)
        for comp, label in zip(self.components, self.labels):
            setattr(self, label, comp)
        self.grid = kwargs.pop("grid", None)
        self.artists = kwargs.pop("artists", [])

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, grid):
        self._grid = grid
        for comp in self.components:
            comp.grid = grid

    @property
    def R(self):
        return self._grid.R

    @property
    def Z(self):
        return self._grid.Z

    @property
    def psi(self):
        return sum([comp.psi for comp in self.components], axis=0)

    @property
    def BR(self):
        return sum([comp.BR for comp in self.components], axis=0)

    @property
    def BZ(self):
        return sum([comp.BZ for comp in self.components], axis=0)

    @property
    def patches(self):
        plist = [c.patches for c in self.components]
        plist = [p for sublist in plist for p in sublist]
        plist.extend(self.artists)
        return plist

    @property
    def patch_coll(self):
        return PatchCollection(self.patches, match_original=True)

    def add_component(self, component, label):
        self.components.append(component)
        self.labels.append(label)
        setattr(self, label, component)

    def reset_grid(self):
        self.grid = self._grid

    def update(self):
        self.reset_grid()
        for comp in self.components:
            comp.update_patches()

    def plot_currents(self, ax):
        for comp in self.components:
            for group in comp.groups:
                group.plot(ax)

    def plot_psi(self, ax, *args, **kwargs):
        return ax.contour(self.grid.R, self.grid.Z, self.psi, *args, **kwargs)

    def plot_modB(self, ax, *args, **kwargs):
        return ax.contour(self.grid.R, self.grid.Z, sqrt(self.BR ** 2 + self.BZ ** 2), *args, **kwargs)

    def plot(self, ax, *args, **kwargs):
        for comp in self.components:
            comp.plot(ax, *args, **kwargs)
        ax.add_collection(PatchCollection(self.artists, match_original=True))

    def save(self, filename=None):
        self.update()
        if filename is None:
            if self.filename is None:
                raise ValueError("I can't find a filename to save to")
            else:
                save_config(self, self.filename)
        else:
            save_config(self, filename)

