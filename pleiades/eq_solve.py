import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

def compute_equilibrium(R,Z,Pfunc,psi_vac,g_psi,tol=1E-10,maxiter=100,relax=0.0,plas_clip=None,plotit=False):
    """ Compute jxB=gradP equilibrium for given P as function of R
    """
    psi0 = psi_vac.flatten()
    psi = psi_vac.flatten()
    psi_old = psi_vac.flatten()
    r = R.flatten()
    z = Z.flatten()
    z0_idx = np.where(z==0.0)
    r_z0 = r[z0_idx]
    dr = np.abs(R[0,1]-R[0,0])
    dz = np.abs(Z[1,0]-Z[0,0])
    p_z0 = np.array(list(map(Pfunc,r_z0)))
    psi_z0 = psi[z0_idx]
    lim_idx = next((i for i, p in enumerate(p_z0) if p == 0.0))
    a = r_z0[lim_idx]
    p_psifit = InterpolatedUnivariateSpline(psi_z0[0:lim_idx+1],p_z0[0:lim_idx+1],ext="zeros")
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
        print(k)
        print(Err) 
        if plotit:
            ## Plot P vs R, Psi vs R, P vs Psi and Pprime vs Psi
            fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex="col")
            ax1.plot(r_z0,list(map(Pfunc,r_z0)),lw=2)
            ax1.set_ylabel("P (Pa)")
            psi_lin = np.linspace(psi_z0.min(),psi_z0.max(),1000)
            ax2.plot(psi_z0,list(map(Pfunc,r_z0)),"o")
            ax2.plot(psi_z0,p_psifit(psi_z0),lw=2)
            ax2.plot(psi_lin,p_psifit(psi_lin),"k--",lw=2)
            ax2.set_ylabel("P (Pa)")
            ax3.plot(r_z0,psi_z0,lw=2)
            ax3.set_ylabel("$\\psi$ (Wb)")
            ax3.set_xlabel("$\r$ (cm)")
            ax4.plot(psi_z0,pprime_fit(psi_z0),"o")
            ax4.set_ylabel("P' (Pa/Wb)")
            ax4.set_xlabel("$\\psi$ (Wb)")
            #plt.tight_layout()
            fig,ax = plt.subplots()
            #psi_levels = np.logspace(np.log10(np.amin(np.abs(psi))),np.log10(np.amax(np.abs(psi))),20)
            ax.set_title("values of psi and currents")
            ax.contour(R,Z,np.abs(psi).reshape(R.shape),tuple(sorted(list(np.linspace(0,np.abs(psi_edge),25)))))
            ax.contourf(R,Z,plas_currents.reshape(R.shape),20)
            fig,ax = plt.subplots()
            ax.set_title("Current error")
            #cf = ax.contourf(R,Z,(psi-psi_old).reshape(R.shape),100)
            cf = ax.contourf(R,Z,(plas_currents-currents_old).reshape(R.shape),100)
            cf1 = ax.contour(R,Z,(psi).reshape(R.shape),100,colors="k")
            plt.colorbar(cf)
            plt.show()

        ## Find pprime everywhere on grid from interpolation and compute
        pprime = pprime_fit(psi)
        pprime[~np.isfinite(pprime)]=0
        pprime[psi > psi_edge]=0
        if plas_clip is not None:
            pprime[plas_clip.flatten()] = 0
        currents_old = np.copy(plas_currents)
        plas_currents = (1-relax)*r*pprime*dr*dz  + relax*currents_old ## in Amps

        psi_plas = np.dot(g_psi,plas_currents)
        psi_plas[~np.isfinite(psi_plas)]=0
        psi_old = np.copy(psi)
        psi = psi_plas + psi0

        ## find new midplane psi vector and edge flux and update p interpolation
        psi_z0 = psi[z0_idx]
        psi_edge = psi_z0[lim_idx]
        p_psifit = InterpolatedUnivariateSpline(psi_z0[0:lim_idx+1],p_z0[0:lim_idx+1],ext="zeros")
        pprime_fit = p_psifit.derivative()

        ## find separatrix and update psi limit
        p_z0 = np.array(list(map(Pfunc,r_z0)))
        psi_z0 = psi[z0_idx]
        lim_idx = next((i for i,p in enumerate(p_z0) if p == 0.0))
        a = r_z0[lim_idx]

        ## Evaluate error between iterations and update counter
        Err = 1./np.abs(psi_edge)*np.sqrt(np.sum((psi - psi_old)**2))
        k+=1
    print(k)
    print(Err)

    return psi.reshape(R.shape),plas_currents.reshape(R.shape),p_psifit

def mirror_equilibrium(brb,gplas,p0,alpha1,alpha2,tol=1E-10,maxiter=100,relax=0.0,plotit=False):
    """ Compute jxB=gradP equilibrium for given P as function of psi/psi_lim given by
    P(psi) = p0*(1-(psi/psi_lim)**alpha1)**alpha2
    """
    R,Z = brb.grid.R, brb.grid.Z
    # force psi_lim through rlim1,zlim1 and rlim2,zlim2
    rlim1,zlim1 = 0.08,0.5
    rlim2,zlim2 = 0.6,0.0
    r1idx,z1idx = np.abs(R[0,:]-rlim1).argmin(), np.abs(Z[:,0]-zlim1).argmin()
    r2idx,z2idx = np.abs(R[0,:]-rlim2).argmin(), np.abs(Z[:,0]-zlim2).argmin()
    gm1,gm2 = np.sum(brb.mirrors.gpsi,axis=-1).reshape(R.shape)[z1idx,r1idx], np.sum(brb.mirrors.gpsi,axis=-1).reshape(R.shape)[z2idx,r2idx]
    gc1,gc2 = np.sum(brb.trex.gpsi,axis=-1).reshape(R.shape)[z1idx,r1idx], np.sum(brb.trex.gpsi,axis=-1).reshape(R.shape)[z2idx,r2idx]
    gp1,gp2 = 0, 0
    iplas = 0
    new_trex_cur = -((gm1-gm2)*brb.mirrors.currents[0] + (gp1-gp2)*iplas)/(gc1 - gc2)
    print(new_trex_cur)
    brb.trex.currents = [new_trex_cur,new_trex_cur]
    psi_vac = brb.psi
    psi_lim = psi_vac[z1idx,r1idx]
    psi_vac_norm = (psi_vac/psi_lim)
    psi_norm = copy.copy(psi_vac_norm)

    dr = np.abs(R[0,1]-R[0,0])
    dz = np.abs(Z[1,0]-Z[0,0])

    plas_currents = np.zeros_like(psi_vac_norm)
    currents_old = np.copy(plas_currents)
    psi_old = np.copy(psi_vac_norm)
    ## Begin iteration
    Err = 1
    k = 0
    xx,_ = np.meshgrid(np.arange(len(R[0,:])),np.arange(len(Z[:,0])))
    while Err > tol and k < maxiter:
        print(k)
        print(Err) 

#        if plotit:
#            ## Plot P vs R, Psi vs R, P vs Psi and Pprime vs Psi
#            fig,((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,sharex="col")
#            ax1.plot(r_z0,list(map(Pfunc,r_z0)),lw=2)
#            ax1.set_ylabel("P (Pa)")
#            psi_lin = np.linspace(psi_z0.min(),psi_z0.max(),1000)
#            ax2.plot(psi_z0,list(map(Pfunc,r_z0)),"o")
#            ax2.plot(psi_z0,p_psifit(psi_z0),lw=2)
#            ax2.plot(psi_lin,p_psifit(psi_lin),"k--",lw=2)
#            ax2.set_ylabel("P (Pa)")
#            ax3.plot(r_z0,psi_z0,lw=2)
#            ax3.set_ylabel("$\\psi$ (Wb)")
#            ax3.set_xlabel("$\r$ (cm)")
#            ax4.plot(psi_z0,pprime_fit(psi_z0),"o")
#            ax4.set_ylabel("P' (Pa/Wb)")
#            ax4.set_xlabel("$\\psi$ (Wb)")
#            plt.tight_layout()
#            fig,ax = plt.subplots()
#            #psi_levels = np.logspace(np.log10(np.amin(np.abs(psi))),np.log10(np.amax(np.abs(psi))),20)
#            ax.set_title("values of psi and currents")
#            ax.contour(R,Z,np.abs(psi).reshape(R.shape),tuple(sorted(list(np.linspace(0,np.abs(psi_edge),25)))))
#            ax.contourf(R,Z,plas_currents.reshape(R.shape),20)
#            fig,ax = plt.subplots()
#            ax.set_title("Current error")
#            #cf = ax.contourf(R,Z,(psi-psi_old).reshape(R.shape),100)
#            cf = ax.contourf(R,Z,(plas_currents-currents_old).reshape(R.shape),100)
#            cf1 = ax.contour(R,Z,(psi).reshape(R.shape),100,colors="k")
#            plt.colorbar(cf)
#            plt.show()

        ## Find pprime everywhere on grid
        pprime = -alpha1*alpha2*p0/psi_lim * (1-psi_norm**alpha1)**(alpha2-1)*psi_norm**(alpha1-1)
        pprime[~np.isfinite(pprime)]=0
        rlimidxs = np.argmax(psi_norm > 1, axis=1)
        plas_mask = xx >= rlimidxs[:,None]
        pprime[plas_mask]=0
        ## Store old plasma currents and compute new ones
        currents_old = np.copy(plas_currents)
        plas_currents = (1-relax)*R*pprime*dr*dz  + relax*currents_old ## in Amps
        cf = plt.contourf(R,Z,plas_currents,101)
        cs = plt.contour(R,Z,psi,51,colors="k",lw=2)
        plt.colorbar(cf)
        plt.gca().set_aspect(1)
        plt.show()

        # update flux from plasma currents
        psi_plas = np.dot(gplas,plas_currents.ravel())
        psi_plas[~np.isfinite(psi_plas)]=0
        psi_old = np.copy(psi)
        psi = psi_plas + psi0

        #find new trex coil current to force same psi_lim shape
        new_trex_cur = -((gm1-gm2)*brb.mirrors.currents[0] + (gp1-gp2).dot(plas_currents.ravel()))/(gc1 - gc2)
        print(new_trex_cur)
        brb.trex.currents = [new_trex_cur,new_trex_cur]
        psi_vac = brb.psi
        #psi = 
        psi_lim = psi_vac[z1idx,r1idx]
        psi_vac_norm = psi_vac/psi_lim

        ## Evaluate error between iterations and update counter
        Err = 1./np.abs(psi_edge)*np.sqrt(np.sum((psi - psi_old)**2))
        k+=1
    print(k)
