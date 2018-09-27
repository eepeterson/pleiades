from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt
from pleiades.helpers import locs_to_vals


# different functions for different physics terms
# also different functions for different maps R-->psi psi-->R


def compute_equilibrium(R, Z, Pfunc, psi_vac, g_psi, bz_vac, br_vac, g_bz, g_br, a0, coil_z, tol=1E-10, maxiter=100,
                        relax=0.0, plas_clip=None, plotit=False):
    """ Compute jxB=gradP equilibrium for given P as function of R
    """
    psi0 = psi_vac.flatten()
    psi = psi_vac.flatten()
    psi_old = psi_vac.flatten()
    r = R.flatten()
    z = Z.flatten()
    z0_idx = np.where(z == 0.0)
    r_z0 = r[z0_idx]
    dr = np.abs(R[0, 1] - R[0, 0])
    dz = np.abs(Z[1, 0] - Z[0, 0])
    p_z0 = np.array([Pfunc(b, a0) for b in r_z0])
    psi_z0 = psi[z0_idx]
    lim_idx = np.argmin(np.abs(r_z0 - a0))
    p_psifit = InterpolatedUnivariateSpline(psi_z0[0:lim_idx + 1], p_z0[0:lim_idx + 1], ext="zeros")
    pprime_fit = p_psifit.derivative()
    pprime = pprime_fit(psi_z0)
    psi_edge = psi_z0[lim_idx]
    plas_currents = np.zeros_like(psi)
    currents_old = np.copy(plas_currents)
    psi_old = np.copy(psi)
    ## Begin iteration
    a_new = a0
    Err = 1
    A_Err = 1
    k = 0
    a_count = 0
    done = False
    while Err > tol and k < maxiter:
        print(k)
        print(Err)
        if plotit:
            ## Plot P vs R, Psi vs R, P vs Psi and Pprime vs Psi
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex="col")
            ax1.plot(r_z0, [Pfunc(r, a0) for r in r_z0], lw=2)
            ax1.set_ylabel("P (Pa)")
            psi_lin = np.linspace(psi_z0.min(), psi_z0.max(), 1000)
            ax2.plot(psi_z0, [Pfunc(r, a0) for r in r_z0], "o")
            ax2.plot(psi_z0, p_psifit(psi_z0), lw=2)
            ax2.plot(psi_lin, p_psifit(psi_lin), "k--", lw=2)
            ax2.set_ylabel("P (Pa)")
            ax3.plot(r_z0, psi_z0, lw=2)
            ax3.set_ylabel("psi (Wb)")
            ax3.set_xlabel("r (cm)")
            ax4.plot(psi_z0, pprime_fit(psi_z0), "o")
            ax4.set_ylabel("P' (Pa/Wb)")
            ax4.set_xlabel("psi (Wb)")
            # plt.tight_layout()
            fig1, ax5 = plt.subplots()
            # psi_levels = np.logspace(np.log10(np.amin(np.abs(psi))),np.log10(np.amax(np.abs(psi))),20)
            ax5.set_title("values of psi and currents")
            ax5.contour(R, Z, np.abs(psi).reshape(R.shape), tuple(sorted(list(np.linspace(0, np.abs(psi_edge), 25)))))
            ax5.contourf(R, Z, plas_currents.reshape(R.shape), 20)
            fig2, ax6 = plt.subplots()
            ax6.set_title("Current error")
            # cf = ax.contourf(R,Z,(psi-psi_old).reshape(R.shape),100)
            cf = ax6.contourf(R, Z, (plas_currents - currents_old).reshape(R.shape), 100)
            cf1 = ax6.contour(R, Z, (psi).reshape(R.shape), 100, colors="k")
            plt.colorbar(cf)
            plt.show()

        if Err < 10 ** -2 and A_Err > 0.01 and done == False:
            ## find separatrix and update psi
            B = np.sqrt((g_br.dot(plas_currents.flatten()).reshape(R.shape) + br_vac) ** 2 + (
                        g_bz.dot(plas_currents.flatten()).reshape(Z.shape) + bz_vac) ** 2)

            indecies = np.unravel_index(B.argmin(), B.shape)
            z_val_sep = Z[indecies[0], 0]
            r_val_sep = R[0, indecies[1]]
            psi_crit_sep = 0.99 * locs_to_vals(R, Z, psi.reshape(R.shape), [[r_val_sep, z_val_sep]])[0]

            z_ind_a = np.argmin(np.abs(Z[:, 0] - 0))
            r_ind_a = np.argmin(np.abs(psi.reshape(R.shape)[z_ind_a, :] - psi_crit_sep))
            if (1 - relax) * R[z_ind_a, r_ind_a] + relax * a_new > a_new:
                a_old = a_new
                a_new = (1 - relax) * R[z_ind_a, r_ind_a] + relax * a_new
                A_Err = np.abs(a_new - a_old)
                a_count = 0
                psi_keep = psi
                plas_currents_keep = plas_currents

        elif done == False:
            a_count += 1

        if a_count > 100 and A_Err > 0.01 and not done:
            done = True
            print("NO LONGER CONVERGING, TURNING TO A PREVIOUS A")
            psi = psi_keep
            plas_currents = plas_currents_keep
            a_new = 0.99 * a_old

        p_z0 = np.array([Pfunc(b, a_new) for b in r_z0])
        psi_z0 = psi[z0_idx]
        lim_idx = np.argmin(np.abs(r_z0 - a_new))
        print(a_new)

        ## find new midplane psi vector and edge flux and update p interpolation
        psi_z0 = psi[z0_idx]
        psi_edge = psi_z0[lim_idx]
        p_psifit = InterpolatedUnivariateSpline(psi_z0[0:lim_idx + 1], p_z0[0:lim_idx + 1], ext="zeros")
        pprime_fit = p_psifit.derivative()

        ## Find pprime everywhere on grid from interpolation and compute
        pprime = pprime_fit(psi)
        pprime[~np.isfinite(pprime)] = 0
        pprime[psi > psi_edge] = 0
        if plas_clip is not None:
            plas_clip = np.logical_or(R ** 2 + Z ** 2 > a_new ** 2, np.abs(Z) > coil_z)
            # plas_clip = np.abs(Z)>coil_z
            pprime[plas_clip.flatten()] = 0
        currents_old = np.copy(plas_currents)
        plas_currents = (1 - relax) * r * pprime * dr * dz + relax * currents_old  ## in Amps

        psi_plas = np.dot(g_psi, plas_currents)
        psi_plas[~np.isfinite(psi_plas)] = 0
        psi_old = np.copy(psi)
        psi = psi_plas + psi0

        '''
        if Err < 10**-2 and A_Err > 0.01 and done == False:
            ## find separatrix and update psi
            B = np.sqrt( (g_br.dot(plas_currents.flatten()).reshape(R.shape) + br_vac)**2 + (g_bz.dot(plas_currents.flatten()).reshape(Z.shape) + bz_vac)**2 )

            indecies = np.unravel_index(B.argmin(), B.shape)
            z_val_sep = Z[indecies[0],0]
            r_val_sep = R[0,indecies[1]]
            psi_crit_sep = 0.99*locs_to_vals(R,Z,psi.reshape(R.shape),[[r_val_sep,z_val_sep]])[0]

            z_ind_a = np.argmin(np.abs(Z[:,0]-0))
            r_ind_a = np.argmin(np.abs(psi.reshape(R.shape)[z_ind_a,:]-psi_crit_sep))
            if (1-relax)*R[z_ind_a,r_ind_a]+relax*a_new > a_new:
                a_old = a_new
                a_new = (1-relax)*R[z_ind_a,r_ind_a]+relax*a_new
                A_Err = np.abs(a_new-a_old)
                a_count =0
                psi_keep = psi
                plas_currents_keep = plas_currents

        elif done == False:
            a_count+=1

        if a_count > 50 and A_Err > 0.01 and not done:
            done = True
            print "NO LONGER CONVERGING, TURNING TO A PREVIOUS A"
            psi = psi_keep
            plas_currents = plas_currents_keep
            a_new = a_old

        p_z0 = np.array([Pfunc(b,a_new) for b in r_z0])
        psi_z0 = psi[z0_idx]
        lim_idx = np.argmin(np.abs(r_z0-a_new))
        print a_new

        ## find new midplane psi vector and edge flux and update p interpolation
        psi_z0 = psi[z0_idx]
        psi_edge = psi_z0[lim_idx]
        p_psifit = InterpolatedUnivariateSpline(psi_z0[0:lim_idx+1],p_z0[0:lim_idx+1],ext="zeros")
        pprime_fit = p_psifit.derivative()
        '''

        ## Evaluate error between iterations and update counter
        Err = 1. / np.abs(psi_edge) * np.sqrt(np.sum((psi - psi_old) ** 2))
        k += 1
    print(k)
    print(Err)

    return psi.reshape(R.shape), plas_currents.reshape(R.shape), Err, a_new
