from warnings import warn, simplefilter
import math
import numpy as np
from scipy.special import ellipk, ellipe
from multiprocessing import Pool, sharedctypes


def compute_greens(rzw, rz_pts):
    """Compute axisymmetric Green's functions for magnetic fields

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
    simplefilter('ignore', RuntimeWarning)

    # Begin calculation of Green's functions based on vector potential
    # psi = R*A_phi from a current loop at r0, z0 on a grid specified by
    # r and z in cylindrical coordinates and with SI units.
    r, z = rz_pts[:,0], rz_pts[:, 1]
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


def compute_greens_mp(rzw, rz_pts):
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
