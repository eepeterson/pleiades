import numpy as np
#import scipy.integrate
from scipy.interpolate import griddata
from scipy.interpolate import splprep,splev
from scipy.optimize import fmin

class FieldLineSet(object):
    def __init__(self,contourset):
        self.flset = {}
        for ilev,lev in enumerate(contourset.levels):
            fllist = [FieldLine(lev,seg) for seg in contourset.allsegs[ilev]]
            self.flset[lev] = fllist 

class FieldLine(object):
    def __init__(self,psi,verts):
        self.psi = psi
        self._verts = verts
        self._interpolate_verts()
        self.reorder_verts()

    def is_closed(self):
        return np.all(self._verts[0,:] == self._verts[-1,:])

    def _interpolate_verts(self,u=None,s=0.0,k=2,npts=1000):
        if self.is_closed():
            per = 1
        else:
            per = 0
        tck,u = splprep(self._verts.T,u=u,k=k,s=s,per=per)
        u_new = np.linspace(u.min(),u.max(),npts)
        self.tck = tck
        self.u = u_new
        r_new,z_new = splev(u_new,tck,der=0)
        self.verts = np.vstack((r_new,z_new)).T

    def interpolate(self,u):
        return splev(u,self.tck,der=0)

    def reorder_verts(self,steptol=0.1):
        print("reordering verts...")
        if not self.is_closed():
            print("Entered if statement...")
#            #old code
#            istart = np.argmin(self.verts[:,0])
#            tmpvert = np.roll(self.verts,-istart,axis=0)
#            if (tmpvert[1,1]-tmpvert[0,1])**2 + (tmpvert[1,0]-tmpvert[0,0])**2 > steptol**2:
#                tmpvert = np.roll(tmpvert,-1,axis=0)[::-1,:]
#            self.verts = tmpvert
            vertdiffs = np.empty_like(self.verts[:,0])
            vertdiffs[1:] = np.sum((self.verts[1:,:]-self.verts[0:-1,:])**2,axis=1)
            vertdiffs[0] = np.sum((self.verts[0,:] - self.verts[-1,:])**2)
            jumpidx = np.nanargmax(vertdiffs)
            self.verts = np.roll(self.verts,-jumpidx,axis=0)
#            if self.verts[-1,1] < self.verts[0,-1]:
#                self.verts = self.verts[::-1,:]

    def get_svec(self):
        s = np.zeros(self.verts.shape[0])
        r,z = self.verts[:,0], self.verts[:,1]
        s[1:] = np.cumsum(np.sqrt((r[1:]-r[0:-1])**2+(z[1:]-z[0:-1])**2))
        return s

    def get_length(self):
        return self.get_svec()[-1]

    def get_ds(self):
        r,z = self.verts[:,0], self.verts[:,1]
        return np.sqrt((r[1]-r[0])**2+(z[1]-z[0])**2)

    def interpolate_onto(self,R,Z,Q,method="cubic"):
        return griddata((R.ravel(),Z.ravel()),Q.ravel(),xi=(self.verts[:,0],self.verts[:,1]),method=method)

    def get_kappa_n(self,R,Z,BR,BZ,method="cubic"):
        modB = np.sqrt(BR**2+BZ**2)
        bhatr, bhatz = BR/modB, BZ/modB
        bhatr_terp = self.interpolate_onto(R,Z,bhatr,method=method)
        bhatz_terp = self.interpolate_onto(R,Z,bhatz,method=method)
        for i in range(self.verts.shape[0]):
            signb = np.sign(self.verts[i,0]*bhatr_terp[i] + self.verts[i,1]*bhatz_terp[i])
            if np.isfinite(signb):
                break
        kap_r, kap_z = signb*self.d_ds(bhatr_terp), signb*self.d_ds(bhatz_terp)
        return np.vstack((kap_r,kap_z)).T

    def d_ds(self,Q):
        ds = self.get_ds()
        res = np.zeros_like(Q)
        res[1:-1] = (Q[2:] - Q[:-2]) / (2*ds)
        res[0] = (-3.0/2.0*Q[0] + 2*Q[1] - 1.0/2.0*Q[2]) / ds
        res[-1] = (1.0/2.0*Q[-3] - 2*Q[-2] + 3.0/2.0*Q[-1]) / ds 
        return res

    def get_gradpsi(self,R,Z,BR,BZ,method="cubic"):
        gradpsi_r = self.interpolate_onto(R,Z,R*BZ,method=method)
        gradpsi_z = -self.interpolate_onto(R,Z,R*BR,method=method)
        return gradpsi_r,gradpsi_z

    def intersection(self,boundary):
        def _distfunc(self,boundary,s1,s2):
            rfl,zfl = self.interpolate(s1)
            rb,zb = boundary.interpolate(s2)
            return (rfl-rb)**2 + (zfl-zb)**2

        distfunc = lambda x0: _distfunc(self,boundary,x0[0],x0[1])
        res = fmin(distfunc,[.5,.5],disp=0)
        return res[0]

    def apply_boundary(self,b1,b2):
        self.ubound0 = self.intersection(b1)
        self.ubound1 = self.intersection(b2)

    def get_bounded_fl(self,npts=1000):
        return self.interpolate(np.linspace(self.ubound0,self.ubound1,npts))
    
def contour_points(contourset):
    condict = {}
    for ilev, lev in enumerate(contourset.levels):
        condict[lev] = [FieldLine(lev,seg) for seg in contourset.allsegs[ilev]]
    return condict

#def flux_surface_avg(Rho,Z,B,flpoints,Q=None):
#    """Compute flux surface average of quantity Q or return dVdpsi (dl_B)
#    """
#    ## Interpolate B and quantity Q onto flux line
#    B_interp = interp(Rho,Z,B,flpoints)
#    s = get_fieldline_distance(flpoints)
#    dl_B = scipy.integrate.trapz(y=1.0/B_interp,x=s)
#    if Q != None:
#        Q_interp = interp(Rho,Z,Q,flpoints)
#        fsa = 1/dl_B*scipy.integrate.trapz(y=Q_interp/B_interp,x=s)
#        return fsa
#    else:
#        return dl_B
#
#def diff_central(x, y):
#    x0 = x[:-2]
#    x1 = x[1:-1]
#    x2 = x[2:]
#    y0 = y[:-2]
#    y1 = y[1:-1]
#    y2 = y[2:]
#    f = (x2 - x1)/(x2 - x0)
#    return (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0)

