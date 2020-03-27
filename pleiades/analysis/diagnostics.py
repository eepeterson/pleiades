import numpy as np

def fit_to_diagnostic(d_mat):
    (u,s,vt) = np.linalg.svd(d_mat)
    Sinv = np.zeros_like(s.T)
    s[ s<1.0e-10 ] = 0.0
    s[ s>=1.0e-10 ] = 1.0/s[ s>=1.0e-10]
    Sinv[:n,:n] = np.diag(s)
    c = vt.T.dot(Sinv.dot(u.T))
    return c

def fit_BR(grid,locs,plas_currents,linear=True):
    # If linear, plasma currents should be R*dR*dZ*P' basismatrix, returns M of BR = M.dot(a)
    # If not linear, plasma currents should be list of plasma currents, returns BR
    gBR = get_greens(grid._R,grid._Z,np.vstack((locs.T,np.ones_like(loc[:,0]))).T)[1] 
    return gBR.dot(plas_currents)

def fit_BZ(grid,locs,plas_currents,linear=True):
    # If linear, plasma currents should be R*dR*dZ*P' basismatrix, returns M of BR = M.dot(a)
    # If not linear plasma currents should be list of plasma currents, returns BR
    gBZ = get_greens(grid._R,grid._Z,np.vstack((locs.T,np.ones_like(loc[:,0]))).T)[2] 
    return gBZ.dot(plas_currents)
