from abc import ABCMeta, abstractmethod
from warnings import warn, simplefilter
import math
import numpy as np
from scipy.special import ellipk, ellipe
from matplotlib.path import Path
import matplotlib.patches as patches


_OUT_OF_DATE_GREENS = """Warning: The greens function for this instance is now
out of date"""

def rotate(pts, angle, pivot=(0., 0.)):
    pivot = np.asarray(pivot)
    angle = math.pi*angle/180
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array([[c, -s], [s, c]])
    return (np.asarray(pts) - pivot) @ rotation + pivot

class CurrentFilamentSet(metaclass=ABCMeta):
    """Set of locations that have the same current value

    Parameters
    ----------
    current : float, optional
        The current to be used for Green's function.
    weights : iterable, optional
        The weights for all the current locations. This enables having both
        positive and negative currents in an object at the same time as well as
        current profile shaping in the case of permanent magnets. Defaults to
        1 for every location.

    Attributes
    ----------
    current : float
        The current in the CurrentGroup in amps.
    weights : iterable
        The weights for all the current locations. This enables having both
        positive and negative currents in an object at the same time as well as
        current profile shaping in the case of permanent magnets. Defaults to
        1 for every location
    rz_pts : iterable
        Nx2 iterable representing R, Z current centroids.
    npts : int
        Integer for the number of current filaments in this object.
    rzw : np.array
        An Nx3 array whos rows are rzw[i, :] = rloc, zloc, weight which
        describe the current location and current weight for each filament in
        the CurrentFilamentSet. This is simply a helper attribute for combining
        rz_pts and weights.
    total_current : float
        The total current being carried in the filament set. This is equal to
        the current times the sum of the weights.
    g_psi : GreensFunction object
        An object representing the Green's function for magnetic flux for this
        CurrentFilamentSet.
    g_br : GreensFunction object
        An object representing the Green's function for the radial component B_R
        for this CurrentFilamentSet.
    g_bz : GreensFunction object
        An object representing the Green's function for the radial component B_R
        for this CurrentFilamentSet.
    """

    def __init__(self, current=1., weights=None):
        self.current = current
        self.weights = weights

    @property
    def current(self):
        return self._current

    @property
    def weights(self):
        return self._weights

    @property
    @abstractmethod
    def rz_pts(self):
        pass

    @property
    @abstractmethod
    def npts(self):
        pass

    @property
    def rzw(self):
        rzw = np.empty((self.npts, 3))
        rzw[:, 0:2] = self.rz_pts
        rzw[:, 2] = self.weights
        return rzw

    @property
    def total_current(self):
        return self.current*np.sum(self.weights)

    @current.setter
    def current(self, current):
        self._current = current

    @weights.setter
    def weights(self, weights):
        print(self.npts)
        if weights is None:
            self._weights = np.ones(self.npts)
        else:
            assert len(weights) == self.npts
            self._weights = np.asarray(weights)

    @abstractmethod
    def translate(self, vector):
        """Translate the current group by the vector (dr, dz)

        Parameters
        ----------
        vector : iterable of float
            The displacement vector for the translation
        """

    @abstractmethod
    def rotate(self, angle, pivot=(0., 0.)):
        """Rotate the current group by a given angle around a specified pivot

        Parameters
        ----------
        angle : float
            The angle of the rotation in degrees as measured from the z axis
        pivot : iterable of float, optional
            The (R, Z) location of the pivot. Defaults to (0., 0.).
        """

    def plot(self, ax, **kwargs):
        """Plot the current locations for the CurrentGroup

        Parameters
        ----------
        ax : matplotlib.Axes object
            The axes object for plotting the current locations
        **kwargs : dict, optional
            Keyword arguments to pass to Current.plot method
        """
        ax.add_patch(self.patch)
        for i, (r, z, w) in enumerate(self.rzw):
            ax.plot(r, z, 'kx', **kwargs)

    def psi(self, grid=None):
        """Compute the magnetic flux, psi, on the desired grid.

        Parameters
        ----------
        grid : np.array
            (R, Z) points at which to calculate the magnetic flux

        Returns
        -------
        psi : np.array
        """
        return self.current*self.g_psi

    def br(self, grid=None):
        """Compute the radial component of the magnetic field, B_R, on the
        desired grid.

        Parameters
        ----------
        grid : np.array
            (R, Z) points at which to calculate the magnetic field Br

        Returns
        -------
        br : np.array
        """
        return self.current*self.g_BR

    def bz(self, grid=None):
        """Compute the radial component of the magnetic field, B_Z, on the
        desired grid.

        Parameters
        ----------
        grid : np.array
            (R, Z) points at which to calculate the magnetic field Bz

        Returns
        -------
        bz : np.array
        """
        return self.current*self.g_BZ

    def compute_greens(self, R, Z, return_greens=False):
        """Helper function for computing Green's functions

        Parameters
        ----------
        R : np.array
            A 1D np.array representing the R positions of the grid
        Z : np.array
            A 1D np.array representing the Z positions of the grid
        """
        simplefilter("ignore", RuntimeWarning)
        R, Z = R.flatten(), Z.flatten()
        n = len(R)
        gpsi = np.zeros(n)
        gBR = np.zeros(n)
        gBZ = np.zeros(n)
        R2 = R**2
        mu_0 = 4*math.pi*1E-7
        pre_factor = mu_0 / (4*math.pi)
        for r0, z0, weight in self.rzw:
            if np.isclose(r0, 0, rtol=0, atol=1E-12):
                continue
            fac0 = (Z - z0)**2
            d = np.sqrt(fac0 + (R + r0)**2)
            d_ = np.sqrt(fac0 + (R - r0)**2)
            k_2 = 4*R*r0 / d**2
            K = ellipk(k_2)
            E = ellipe(k_2)
            denom = d*d_ **2
            fac1 = K*d_ **2
            fac2 = (fac0 + R2 + r0**2)*E
            gpsi_tmp = weight*pre_factor*R*r0*4 / d / k_2*((2 - k_2)*K - 2*E)
            gpsi_tmp[~np.isfinite(gpsi_tmp)] = 0
            gpsi += gpsi_tmp
            gBR_tmp = -2*weight*pre_factor*(Z - z0)*(fac1 - fac2) / (R*denom)
            gBR_tmp[~np.isfinite(gBR_tmp)] = 0
            gBR += gBR_tmp
            gBZ_tmp = 2*weight*pre_factor*(fac1 - (fac2 - 2*r0**2*E)) / denom
            gBZ_tmp[~np.isfinite(gBZ_tmp)] = 0
            gBZ += gBZ_tmp

        self.g_psi = gpsi
        self.g_BR = gBR
        self.g_BZ = gBZ
        if return_greens:
            return gpsi, gBR, gBZ
