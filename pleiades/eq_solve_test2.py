import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from fields.helpers import locs_to_vals
import copy

import plottingtools.plottingtools as ptools

import copy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

# different functions for different physics terms
# also different functions for different maps R-->psi psi-->R

cmap_r = ptools.truncate_colormap(plt.get_cmap('CMRmap_r'), minval=.05, maxval=.85)
cmap = ptools.truncate_colormap(plt.get_cmap('CMRmap'), minval=.15, maxval=.95)


def compute_equilibrium(R, Z, Pfunc, psi_vac, g_psi, bz_vac, br_vac, g_bz, g_br, a0, coil_z, tol=1E-10, maxiter=100,
                        relax=0.0, plas_clip=None, plotit=False):
    """ Compute jxB=gradP equilibrium for given P as function of R
    """

    A_err = 1
    a_new = a0
    while A_err > 0:

        psi0 = psi_vac.flatten()
        psi = psi_vac.flatten()
        psi_old = psi_vac.flatten()
        r = R.flatten()
        z = Z.flatten()
        z0_idx = np.where(z == 0.0)
        r_z0 = r[z0_idx]
        dr = np.abs(R[0, 1] - R[0, 0])
        dz = np.abs(Z[1, 0] - Z[0, 0])

        p_z0 = np.array([Pfunc(b, a_new) for b in r_z0])

        psi_z0 = psi[z0_idx]
        lim_idx = (i for i, p in enumerate(p_z0) if p == 0.0).next()
        a = r_z0[lim_idx]
        p_psifit = InterpolatedUnivariateSpline(psi_z0[0:lim_idx + 1], p_z0[0:lim_idx + 1], ext="zeros")
        pprime_fit = p_psifit.derivative()
        pprime = pprime_fit(psi_z0)
        psi_edge = psi_z0[lim_idx]
        plas_currents = np.zeros_like(psi)
        currents_old = np.copy(plas_currents)
        psi_old = np.copy(psi)
        ## Begin iteration
        Err = 1
        k = 0
        while Err > tol and k < maxiter:
            print k
            print Err
            if plotit:
                ## Plot P vs R, Psi vs R, P vs Psi and Pprime vs Psi
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="col")
                ax1.plot(r_z0, map(Pfunc, r_z0), lw=2)
                ax1.set_ylabel("P (Pa)")
                psi_lin = np.linspace(psi_z0.min(), psi_z0.max(), 1000)
                ax2.plot(psi_z0, map(Pfunc, r_z0), "o")
                ax2.plot(psi_z0, p_psifit(psi_z0), lw=2)
                ax2.plot(psi_lin, p_psifit(psi_lin), "k--", lw=2)
                ax2.set_ylabel("P (Pa)")
                ax3.plot(r_z0, psi_z0, lw=2)
                ax3.set_ylabel("$\\psi$ (Wb)")
                ax3.set_xlabel("$\r$ (cm)")
                ax4.plot(psi_z0, pprime_fit(psi_z0), "o")
                ax4.set_ylabel("P' (Pa/Wb)")
                ax4.set_xlabel("$\\psi$ (Wb)")
                plt.tight_layout()
                fig, ax = plt.subplots()
                # psi_levels = np.logspace(np.log10(np.amin(np.abs(psi))),np.log10(np.amax(np.abs(psi))),20)
                ax.set_title("values of psi and currents")
                ax.contour(R, Z, np.abs(psi).reshape(R.shape),
                           tuple(sorted(list(np.linspace(0, np.abs(psi_edge), 25)))))
                ax.contourf(R, Z, plas_currents.reshape(R.shape), 20)
                fig, ax = plt.subplots()
                ax.set_title("Current error")
                # cf = ax.contourf(R,Z,(psi-psi_old).reshape(R.shape),100)
                cf = ax.contourf(R, Z, (plas_currents - currents_old).reshape(R.shape), 100)
                cf1 = ax.contour(R, Z, (psi).reshape(R.shape), 100, colors="k")
                plt.colorbar(cf)
                plt.show()

            ## Find pprime everywhere on grid from interpolation and compute
            pprime = pprime_fit(psi)
            pprime[~np.isfinite(pprime)] = 0
            pprime[psi > psi_edge] = 0
            if plas_clip is not None:
                pprime[plas_clip.flatten()] = 0
            currents_old = np.copy(plas_currents)
            plas_currents = (1 - relax) * r * pprime * dr * dz + relax * currents_old  ## in Amps

            psi_plas = np.dot(g_psi, plas_currents)
            psi_plas[~np.isfinite(psi_plas)] = 0
            psi_old = np.copy(psi)
            psi = psi_plas + psi0

            ## find new midplane psi vector and edge flux and update p interpolation
            psi_z0 = psi[z0_idx]
            psi_edge = psi_z0[lim_idx]
            p_psifit = InterpolatedUnivariateSpline(psi_z0[0:lim_idx + 1], p_z0[0:lim_idx + 1], ext="zeros")
            pprime_fit = p_psifit.derivative()

            ## Evaluate error between iterations and update counter
            Err = 1. / np.abs(psi_edge) * np.sqrt(np.sum((psi - psi_old) ** 2))
            k += 1
        print k
        print Err

        # will crash here if only run once cause referenced before assignment
        if Err > tol:
            print "Final A = " + str(a_old)
            return psi_return_old.reshape(R.shape), plas_currents_return_old.reshape(R.shape), Err_old

        B = np.sqrt((g_br.dot(plas_currents.flatten()).reshape(R.shape) + br_vac) ** 2 + (
                    g_bz.dot(plas_currents.flatten()).reshape(Z.shape) + bz_vac) ** 2)

        indecies = np.unravel_index(B.argmin(), B.shape)
        z_val_sep = Z[indecies[0], 0]
        r_val_sep = R[0, indecies[1]]
        psi_crit_sep = 0.99 * locs_to_vals(R, Z, psi.reshape(R.shape), [[r_val_sep, z_val_sep]])[0]

        z_ind_a = np.argmin(np.abs(Z[:, 0] - 0))
        r_ind_a = np.argmin(np.abs(psi.reshape(R.shape)[z_ind_a, :] - psi_crit_sep))
        A_err = np.abs(R[z_ind_a, r_ind_a] - a_new)

        a_old = a_new
        a_new = (1 - relax) * R[z_ind_a, r_ind_a] + relax * a_new
        plas_clip = np.logical_or(R ** 2 + Z ** 2 > a_new ** 2, np.abs(Z) > coil_z)

        print A_err, a_new

        psi_return_old = copy.copy(psi)
        plas_currents_return_old = copy.copy(plas_currents)
        Err_old = Err

    return psi.reshape(R.shape), plas_currents.reshape(R.shape), Err
