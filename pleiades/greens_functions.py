import math
#from numpy import (pi, linspace, meshgrid, sin, cos, sqrt, sum, array, ones,
#                   zeros, hstack, vstack, sign, mod, isfinite, ceil, isclose)
import numpy as np
from scipy.special import ellipk, ellipe
from multiprocessing import Process, Queue, cpu_count
from numbers import Number
from warnings import warn, simplefilter
from abc import ABCMeta

#def _get_greens(R, Z, rzdir, out_q=None):
#    """Helper function for computing Green's functions
#
#    Parameters
#    ----------
#    R : np.array
#        A 1D np.array representing the R positions of the grid
#    Z : np.array
#        A 1D np.array representing the Z positions of the grid
#    rzdir : np.array
#        An Nx3 np.array representing the r, z positions and sign of the current
#        in all the current filaments.
#    out_q: multiprocessing.Queue object?
#        Internally used for faster processing of Green's functions
#    """
#    simplefilter("ignore", RuntimeWarning)
#    n = len(R)
#    gpsi = zeros(n)
#    gBR = zeros(n)
#    gBZ = zeros(n)
#    R2 = R ** 2
#    mu_0 = 4 * pi * 10 ** -7
#    pre_factor = mu_0 / (4 * pi)
#    for r0, z0, csign in rzdir:
#        if isclose(r0, 0, rtol=0, atol=1E-12):
#            continue
#        fac0 = (Z - z0) ** 2
#        d = sqrt(fac0 + (R + r0) ** 2)
#        d_ = sqrt(fac0 + (R - r0) ** 2)
#        k_2 = 4 * R * r0 / d ** 2
#        K = ellipk(k_2)
#        E = ellipe(k_2)
#        denom = d_ ** 2 * d
#        fac1 = d_ ** 2 * K
#        fac2 = (fac0 + R2 + r0 ** 2) * E
#        gpsi_tmp = csign * pre_factor * R * r0 / d * 4 / k_2 * ((2 - k_2) * K - 2 * E)
#        gpsi_tmp[~isfinite(gpsi_tmp)] = 0
#        gpsi += gpsi_tmp
#        gBR_tmp = -2 * csign * pre_factor * (Z - z0) * (fac1 - fac2) / (R * denom)
#        gBR_tmp[~isfinite(gBR_tmp)] = 0
#        gBR += gBR_tmp
#        gBZ_tmp = 2 * csign * pre_factor * (fac1 - (fac2 - 2 * r0 ** 2 * E)) / denom
#        gBZ_tmp[~isfinite(gBZ_tmp)] = 0
#        gBZ += gBZ_tmp
#    out_tup = (gpsi, gBR, gBZ)
#    if out_q is None:
#        return out_tup
#    out_q.put(out_tup)
#
#
#def get_greens(R, Z, rzdir, out_q=None, out_idx=None):
#    """Compute Green's functions for psi, BR, and BZ
#
#    Parameters
#    ----------
#    R : np.array
#        A 1D np.array representing the R positions of the grid
#    Z : np.array
#        A 1D np.array representing the Z positions of the grid
#    rzdir : np.array
#        An Nx3 np.array representing the r, z positions and sign of the current
#        in all the current filaments.
#    out_q: multiprocessing.Queue object?
#        Internally used for faster processing of Green's functions
#    out_idx: int?
#        Internally used for faster processing of Green's functions
#
#    Returns
#    -------
#    out_tup : tuple
#        Tuple of 3 elements (gpsi, gBR, gBZ) for the 3 Green's functions
#    """
#    simplefilter("ignore", RuntimeWarning)
#    m, n = len(R), len(rzdir)
#    gpsi = zeros((m, n))
#    gBR = zeros((m, n))
#    gBZ = zeros((m, n))
#    R2 = R ** 2
#    mu_0 = 4 * pi * 10 ** -7
#    pre_factor = mu_0 / (4 * pi)
#    for i, (r0, z0, csign) in enumerate(rzdir):
#        if isclose(r0, 0, rtol=0, atol=1E-12):
#            continue
#        fac0 = (Z - z0) ** 2
#        d = sqrt(fac0 + (R + r0) ** 2)
#        d_ = sqrt(fac0 + (R - r0) ** 2)
#        k_2 = 4 * R * r0 / d ** 2
#        K = ellipk(k_2)
#        E = ellipe(k_2)
#        denom = d_ ** 2 * d
#        fac1 = d_ ** 2 * K
#        fac2 = (fac0 + R2 + r0 ** 2) * E
#        gpsi_tmp = csign * pre_factor * R * r0 / d * 4 / k_2 * ((2 - k_2) * K - 2 * E)
#        gpsi_tmp[~isfinite(gpsi_tmp)] = 0
#        gpsi[:, i] = gpsi_tmp
#        gBR_tmp = -2 * csign * pre_factor * (Z - z0) * (fac1 - fac2) / (R * denom)
#        gBR_tmp[~isfinite(gBR_tmp)] = 0
#        gBR[:, i] = gBR_tmp
#        gBZ_tmp = 2 * csign * pre_factor * (fac1 - (fac2 - 2 * r0 ** 2 * E)) / denom
#        gBZ_tmp[~isfinite(gBZ_tmp)] = 0
#        gBZ[:, i] = gBZ_tmp
#    out_tup = (gpsi, gBR, gBZ)
#    if out_q is None:
#        return out_tup
#    else:
#        if out_idx is None:
#            raise ValueError("I don't know where to put this output, please specify out_idx")
#        out_q.put((out_idx,) + out_tup)
#
#
#def compute_greens(R, Z, rzdir=None, nprocs=1):
#    """Compute Green's functions for psi, BR, and BZ
#
#    Parameters
#    ----------
#    R : np.array
#        A 1D np.array representing the R positions of the grid
#    Z : np.array
#        A 1D np.array representing the Z positions of the grid
#    rzdir : np.array
#        An Nx3 np.array representing the r, z positions and sign of the current
#        in all the current filaments.
#    out_q: multiprocessing.Queue object?
#        Internally used for faster processing of Green's functions
#    out_idx: int?
#        Internally used for faster processing of Green's functions
#
#    Returns
#    -------
#    out_tup : tuple
#        Tuple of 3 elements (gpsi, gBR, gBZ) for the 3 Green's functions
#    """
#    simplefilter("ignore", RuntimeWarning)
#    proc_max = cpu_count()
#    if rzdir is None:
#        rzdir = vstack((R, Z, ones(len(R)))).T
#    m, n = len(R), len(rzdir)
#    gpsi = zeros((m, n))
#    gBR = zeros((m, n))
#    gBZ = zeros((m, n))
#    if nprocs > proc_max:
#        nprocs = proc_max
#    procs = []
#    out_q = Queue()
#    chunksize = int(ceil(rzdir.shape[0] / float(nprocs)))
#    print(chunksize)
#    for i in range(nprocs):
#        p = Process(target=get_greens, args=(R, Z, rzdir[i * chunksize:(i + 1) * chunksize, :]),
#                    kwargs={"out_q": out_q, "out_idx": i})
#        procs.append(p)
#        p.start()
#
#    for j in range(nprocs):
#        print("getting g_tup #: {0}".format(j))
#        g_tup = out_q.get()
#        idx = g_tup[0]
#        gpsi[:, idx * chunksize:(idx + 1) * chunksize] = g_tup[1]
#        gBR[:, idx * chunksize:(idx + 1) * chunksize] = g_tup[2]
#        gBZ[:, idx * chunksize:(idx + 1) * chunksize] = g_tup[3]
#
#    for p in procs:
#        p.join()
#
#    return (gpsi, gBR, gBZ)
#
