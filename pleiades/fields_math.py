from warnings import warn, simplefilter
import math
import numpy as np
from scipy.special import ellipk, ellipe


def compute_greens(self, rzw, rz_pts):
    """Helper function for computing Green's functions

    Parameters
    ----------
    rzw: Nx3 np.array
        An Nx3 array whose columns are r locations, z locations, and current
        weights respectively for the current filaments.
    rz_pts: Nx2 np.array
        An Nx2 array whose columns are r locations and z locations for the grid
        points where we want to calculate the Green's functions.

    Returns
    -------
    tuple :
        3-tuple of 1D np.array representing the Green's function for psi, BR,
        and Bz respectively.
    """

    # Begin calculation of Green's functions based on vector potential
    # psi = R*A_phi from a current loop at r0, z0 on a grid specified by
    # r and z in cylindrical coordinates and with SI units.
    r, z = rz_pts[:,0], rz_pts[:, 1]
    n = len(r)
    gpsi = np.zeros(n)
    gBR = np.zeros(n)
    gBZ = np.zeros(n)
    r2 = r**2

    # Prefactor c1 for vector potential is mu_0/4pi = 1E-7
    c1 = 1E-7
    for r0, z0, wgt in rzw:
        # Check if the coil position is close to 0 if so skip it
        if np.isclose(r0, 0, rtol=0, atol=1E-12):
            continue

        # Compute factors that are reused in equations
        fac0 = (z - z0)**2
        d = np.sqrt(fac0 + (r + r0)**2)
        d_ = np.sqrt(fac0 + (r - r0)**2)
        k_2 = 4*r*r0 / d**2
        K = ellipk(k_2)
        E = ellipe(k_2)
        denom = d*d_ **2
        fac1 = K*d_ **2
        fac2 = (fac0 + r2 + r0**2)*E

        # Compute Green's functions for psi, BR, BZ
        gpsi_tmp = wgt*c1*r*r0*4 / d / k_2*((2 - k_2)*K - 2*E)
        gBR_tmp = -2*wgt*c1*(z - z0)*(fac1 - fac2) / (r*denom)
        gBZ_tmp = 2*wgt*c1*(fac1 - (fac2 - 2*r0**2*E)) / denom

        # Correct for infinities and add sum
        gpsi_tmp[~np.isfinite(gpsi_tmp)] = 0
        gpsi += gpsi_tmp
        gBR_tmp[~np.isfinite(gBR_tmp)] = 0
        gBR += gBR_tmp
        gBZ_tmp[~np.isfinite(gBZ_tmp)] = 0
        gBZ += gBZ_tmp

    return gpsi, gBR, gBZ
