import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline,UnivariateSpline,griddata
from scipy.spatial import Delaunay, ConvexHull
from matplotlib.collections import LineCollection
from pleiades.math import get_gpsi
from scipy.interpolate import splprep,splev
from scipy.optimize import fmin
from matplotlib.path import Path
#import analysis.datahelpers as dh

class Boundary(object):
    def __init__(self,vertices):
        self._interpolate_verts(vertices)

    def _interpolate_verts(self,vertices,u=None,s=0.0,npts=200):
        tck,u = splprep(vertices.T,u=u,s=s)
        u_new = np.linspace(u.min(),u.max(),npts)
        self.tck = tck
        self.u = u_new
        r_new,z_new = splev(u_new,tck,der=0)
        self.verts = np.vstack((r_new,z_new)).T

    def interpolate(self,u):
        return splev(u,self.tck,der=0)

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
        if not self.is_closed():
            istart = np.argmin(self.verts[:,0])
            tmpvert = np.roll(self.verts,-istart,axis=0)
            if (tmpvert[1,1]-tmpvert[0,1])**2 + (tmpvert[1,0]-tmpvert[0,0])**2 > steptol**2:
                tmpvert = np.roll(tmpvert,-1,axis=0)[::-1,:]
            self.verts = tmpvert

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
        bhatr_terp = self.interpolate_onto(R,Z,bhatr)
        bhatz_terp = self.interpolate_onto(R,Z,bhatz)
        signb = np.sign(self.verts[0,0]*bhatr_terp[0] + self.verts[0,1]*bhatz_terp[0])
        kap_r, kap_z = signb*self.d_ds(bhatr_terp), signb*self.d_ds(bhatz_terp)
        return kap_r, kap_z

    def d_ds(self,Q):
        ds = self.get_ds()
        res = np.zeros_like(Q)
        res[1:-1] = (Q[2:] - Q[:-2]) / (2*ds)
        res[0] = (-3.0/2.0*Q[0] + 2*Q[1] - 1.0/2.0*Q[2]) / ds
        res[-1] = (1.0/2.0*Q[-3] - 2*Q[-2] + 3.0/2.0*Q[-1]) / ds 
        return res

    def get_gradpsi(self,R,Z,BR,BZ,method="cubic"):
        gradpsi_r = self.interpolate_onto(R,Z,R*BZ)
        gradpsi_z = -self.interpolate_onto(R,Z,R*BR)
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

def regular_grid(xx,yy,*args,**kwargs):
    nx = kwargs.pop("nx",200)
    ny = kwargs.pop("ny",200)
    xi = kwargs.pop("xi",None)
    yi = kwargs.pop("yi",None)
    method = kwargs.pop("method","linear")
    """ interpolate irregularly gridded data onto regular grid."""
    if xi is not None and yi is not None:
        pass
    else:
        x = np.linspace(xx.min(), xx.max(), nx)
        y = np.linspace(yy.min(), yy.max(), ny)
        xi, yi = np.meshgrid(x,y,indexing="ij")

    #then, interpolate your data onto this grid:
    points = np.vstack((xx.flatten(),yy.flatten())).T
    zs = []
    for z in args:
        zi = griddata(points,z.flatten(),(xi,yi),method=method)
        zs.append(zi)
    
    return (xi,yi) + tuple(zs)

def get_deltapsi(data,Req,Zeq):
    """ Returns contribution to psi from fast ion currents.

    Args:
        data (netcdf4 Dataset object)
        Req (2D R grid from eqdsk)
        Zeq (2D Z grid from eqdsk)

    Returns:
        deltapsi (psi from fast ion currents on eqdsk grid)
    """
    var = data.variables
    dims = data.dimensions
    nrj = dims["nreqadim"].size
    nzj = dims["nzeqadim"].size
    req = np.linspace(np.min(Req),np.max(Req),nrj)
    zeq = np.linspace(np.min(Zeq),np.max(Zeq),nzj)
    dr,dz = req[1]-req[0], zeq[1]-zeq[0]
    rr,zz = np.meshgrid(req,zeq)
    gpsi_jphi = get_gpsi(rr,zz)
    jphi = var["curr_diamcurv_phi"][:]
    if len(jphi.shape) > 2:
        jphi = np.sum(jphi,axis=0)
    jphi*=1E4 # A/cm^2 to A/m^2
    Iphi = jphi*dr*dz
    deltapsi = (gpsi_jphi.dot(Iphi.flatten())).reshape(rr.shape)
    _,_,deltapsi = regular_grid(rr,zz,deltapsi,xi=Req,yi=Zeq)
    return deltapsi

def poly_fit(x,y,order=3):
    n = order+1
    m = len(y)
    basis_fns = [(lambda z,i=i: z**i) for i in range(n)]
    A = np.zeros((m,n))
    for i in range(m):
        for j in range(n):
            A[i,j] = basis_fns[j](x[i])
    (u,s,vt) = np.linalg.svd(A)
    Sinv = np.zeros((n,m))
    s[ s<1.0e-10 ] = 0.0
    s[ s>=1.0e-10 ] = 1.0/s[ s>=1.0e-10]
    Sinv[:n,:n] = np.diag(s)
    c = np.dot(Sinv,u.T)
    c = np.dot(vt.T,c)
    c = np.dot(c,y)
    return basis_fns,c

def reflect_and_hstack(Rho,Z,*args):
    """Reflect and concatenate grid and quantities in *args to plot both half planes (rho>=0 and rho<=0).

    Note: Currently this function only reflects across the z axis since that is the symmetry convention
        we've taken for the machine.

    Args:
        Rho (2D array): rho grid
        Z (2D array): z grid
        *args (2D arrays), any quantity on this grid you wish to plot in both half planes
    """
    Rho_total = np.hstack((-Rho[:,-1:0:-1],Rho))
    Z_total = np.hstack((Z[:,-1:0:-1],Z))
    arglist = []
    for arg in args:
        assert arg.shape == Rho.shape
        arglist.append(np.hstack((arg[:,-1:0:-1],arg)))
    return (Rho_total,Z_total)+tuple(arglist)

def get_concave_hull(Rho,Z,Q):
    points = np.array([[rho0,z0] for rho0,z0,q in zip(Rho.flatten(),Z.flatten(),Q.flatten()) if ~np.isnan(q)])
    tri = Delaunay(points)

    # Make a list of line segments: 
    # edge_points = [ ((x1_1, y1_1), (x2_1, y2_1)),
    #                 ((x1_2, y1_2), (x2_2, y2_2)),
    #                 ... ]
    edge_points = []
    edges = set()

    def add_edge(i, j):
        """Add a line between the i-th and j-th points, if not in the list already"""
        if (i, j) in edges or (j, i) in edges:
        # already added
            return
        edges.add( (i, j) )
        edge_points.append(points[ [i, j] ])

    # loop over triangles: 
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        add_edge(ia, ib)
        add_edge(ib, ic)
        add_edge(ic, ia)

    # plot it: the LineCollection is just a (maybe) faster way to plot lots of
    # lines at once

    lines = LineCollection(edge_points)
    plt.figure()
    plt.title('Delaunay triangulation')
    plt.gca().add_collection(lines)
    #plt.plot(points[:,0], points[:,1], 'o', hold=1)
    plt.xlim(-20, 20)
    plt.ylim(-10, 10)
    plt.show()

def transform_to_rtheta(Rho,Z,rho_component,z_component):
    """Transform Rho Z grid and rho,z components of vector field to polar coordinates"""
    R = np.sqrt(Rho**2+Z**2)
    Theta = np.pi/2+np.arctan2(Z,Rho)
    sin_t = np.sin(Theta)
    cos_t = np.cos(Theta)
    r_component = rho_component*sin_t + z_component*cos_t
    theta_component = rho_component*cos_t - z_component*sin_t
    return R,Theta,r_component,theta_component 

def transform_to_rhoz(R,Theta,r_component,theta_component):
    """Transform R Theta grid and r theta components of vector field to cylindrical coordinates"""
    Rho = R*np.sin(Theta)
    Z = R*np.cos(Theta)
    rho_component = (r_component*Rho + theta_component*Z)/R
    z_component = (r_component*Z - theta_component*Rho)/R
    return Rho,Z,rho_component,z_component 

def locs_to_vals(X,Y,Q,coord_list):
    """Picks values of field Q at desired coordinates.

    Args:
        X (2D array): grid representing column coordinates of Q
        Y (2D array): grid representing row coordinates of Q
        Q (2D array): value of Q on grid
        coord_list (list):list of tuples (x,y) for desired coordinates
    """
    q_vals = []
    for x,y in coord_list:
        x0_idx,y0_idx = np.argmin(np.abs(X[0,:]-x)),np.argmin(np.abs(Y[:,0]-y))
        q_vals.append(Q[y0_idx,x0_idx])
    return q_vals

def locs_to_vals_griddata(X,Y,Q,coord_list):
    """Picks values of field Q at desired coordinates.

    Args:
        X (2D array): grid representing column coordinates of Q
        Y (2D array): grid representing row coordinates of Q
        Q (2D array): value of Q on grid
        coord_list (list):list of tuples (x,y) for desired coordinates
    """
    xi,yi = zip(*coord_list)
    return griddata((X.flatten(),Y.flatten()),Q.flatten(),(xi,yi))

def locs_to_vals1D(X,Y,Q,coord_list):
    """Picks values of field Q at desired coordinates.

    Args:
        X (2D array): grid representing column coordinates of Q
        Y (2D array): grid representing row coordinates of Q
        Q (2D array): value of Q on grid
        coord_list (list):list of tuples (x,y) for desired coordinates
    """
    q_vals = []
    for x,y in coord_list:
        idx = ((X-x)**2 + (Y-y)**2).argmin()
        q_vals.append(Q[idx])
    return q_vals

def get_fieldlines(contourset,level,start_coord=None,end_coord=None,clockwise=True,idx_check=[]):
    """Return coordinates for segments comprising a flux surface (Nx2 array).

    Args:
        contourset (matplotlib.contour.QuadContourSet instance): i.e.
        ax.contour call
        level (float): desired contour level
        start_coord (tuple): coordinate (x,y) at which to start the field line
        end_coord (tuple): coordinate (x,y) at which to end the field line
        clockwise (bool): whether to order the field line coordinates clockwise or
        counterclockwise
    """
    ## Find desired flux surface and get its vertices
    assert level in list(contourset.levels), "level: {0} not found in contourset".format(level)
    idx = list(contourset.levels).index(level)
    segs = contourset.allsegs[idx]
    len_list = [s.shape[0] for s in segs]
    max_idx = len_list.index(max(len_list))
    flpoints = parse_segment(segs[max_idx],start_coord=start_coord,end_coord=end_coord,clockwise=clockwise)
#    if idx in idx_check:
#        fig_pts,ax_pts = plt.subplots()
#        fig_B,ax_B = plt.subplots()
#        for i,pts in enumerate(segs):
#            tmp_pts = parse_segment(pts,start_coord=start_coord,end_coord=end_coord,clockwise=clockwise)
#            ax_pts.plot(tmp_pts[:,0],tmp_pts[:,1],"o")
#            ax_pts.plot(tmp_pts[0,0],tmp_pts[0,1],"go")
#            ax_pts.plot(tmp_pts[-1,0],tmp_pts[-1,1],"ro")
#            ax_pts.set_xlim(0,2.50)
#            ax_pts.set_ylim(-1.25,1.25)
#            ax_pts.set_aspect(1)
#            B_interp = interp(Rho,Z,B,tmp_pts)
#            s = get_fieldline_distance(tmp_pts)
#            ax_B.plot(s,B_interp,"o")
#        plt.show()
    return flpoints

def parse_segment(flpoints,start_coord=None,end_coord=None,clockwise=True):
    if start_coord != None:
        i_start = np.argmin(np.array([x0**2+y0**2 for x0,y0 in zip(flpoints[:,0]-start_coord[0],flpoints[:,1]-start_coord[1])]))
        flpoints = np.roll(flpoints,-i_start,axis=0)
    ## Find out if curve is cw or ccw
    x = flpoints[:,0]
    y = flpoints[:,1]
    iscw = np.sum((x[1:]-x[0:-1])*(y[1:]+y[0:-1]))+(x[0]-x[-1])*(y[0]+y[-1]) > 0
    if clockwise != iscw:
        flpoints = np.roll(flpoints[::-1,:],1,axis=0)
    i_end = len(x)-1
    if end_coord != None:
        i_end = np.argmin(np.array([x0**2+y0**2 for x0,y0 in zip(flpoints[:,0]-end_coord[0],flpoints[:,1]-end_coord[1])]))
        if i_end < len(x)-1:
            i_end += 1
    flpoints = flpoints[0:i_end,:]
    return flpoints

def get_fieldline_distance(flpoints):
    """Return cumulative field line distance vector
    """
    s = np.zeros(flpoints.shape[0])
    x = flpoints[:,0]
    y = flpoints[:,1]
    s[1:] = np.cumsum(np.sqrt((x[1:]-x[0:-1])**2+(y[1:]-y[0:-1])**2))
    return s

def interp(Rho,Z,Q,flpoints):
    """interpolate quantity Q on Rho, Z grid onto flpoints (Nx2 array of x,y pairs).
    """
    x0 = Rho[0,:].squeeze()
    y0 = Z[:,0].squeeze()
    f = RectBivariateSpline(y0,x0,Q)
    return np.array([float(f(yi,xi)[0]) for xi,yi in zip(flpoints[:,0],flpoints[:,1])])

def flux_surface_avg(Rho,Z,B,flpoints,Q=None):
    """Compute flux surface average of quantity Q or return dVdpsi (dl_B)
    """
    ## Interpolate B and quantity Q onto flux line
    B_interp = interp(Rho,Z,B,flpoints)
    s = get_fieldline_distance(flpoints)
    dl_B = scipy.integrate.trapz(y=1.0/B_interp,x=s)
    if Q != None:
        Q_interp = interp(Rho,Z,Q,flpoints)
        fsa = 1/dl_B*scipy.integrate.trapz(y=Q_interp/B_interp,x=s)
        return fsa
    else:
        return dl_B

def diff_central(x, y):
    x0 = x[:-2]
    x1 = x[1:-1]
    x2 = x[2:]
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    f = (x2 - x1)/(x2 - x0)
    return (1-f)*(y2 - y1)/(x2 - x1) + f*(y1 - y0)/(x1 - x0)

# need to remove datahelpers dependency from this before using
#def get_F(Rho,Z,psi,B,min_rho,max_rho,start_coord=None,end_coord=None,clockwise=True,plotit=False,dfl_tol=.1):
#    gamma = 5.0/3.0
#    figf,axf = plt.subplots()
#    psi_min,psi_edge = locs_to_vals(Rho,Z,psi,[(min_rho,0),(max_rho,0)])
#    psi_levels = np.linspace(psi_min,psi_edge,500)
#    cff = axf.contour(Rho,Z,psi,tuple(psi_levels))
#    dl_B_list = []
#    for psi0 in psi_levels:
#        if psi0 == 0.0:
#            flpoints = get_fieldlines(cff,psi0,start_coord=start_coord,end_coord=end_coord,clockwise=False)
#        else:
#            flpoints = get_fieldlines(cff,psi0,start_coord=start_coord,end_coord=end_coord,clockwise=clockwise)
#        s = get_fieldline_distance(flpoints)
#        if np.max(s[1:]-s[0:-1]) > dfl_tol:
#            raise ValueError("fieldline distance between successive points greater than dfl_tol: {0}m".format(dfl_tol))
#        if plotit:
#            x,y = flpoints[:,0],flpoints[:,1]
#            axf.plot(x,y,'bo')
#            axf.plot(x[0],y[0],'go')
#            axf.plot(x[-1],y[-1],'ro')
#        dl_B_list.append(flux_surface_avg(Rho,Z,B,flpoints))
#    psi_new = psi_levels
#    dl_B_new = np.array(dl_B_list)
#    dl_B_new = dh.smooth(dl_B_new,5,mode="valid")
#    psi_new = dh.smooth(psi_new,5,mode="valid")
#    U = UnivariateSpline(psi_new,dl_B_new,k=4,s=0)
#    dUdpsi = UnivariateSpline(psi_new[1:-1],diff_central(psi_new,dl_B_new),k=1,s=0,ext="const")
#    d2Udpsi2 = UnivariateSpline(psi_new[2:-2],diff_central(psi_new[1:-1],diff_central(psi_new,dl_B_new)),k=1,s=0,ext="const")
#    #dUdpsi = UnivariateSpline(psi_new[1:-1],diff_central(psi_new,dl_B_new),k=3,s=0,ext="const")
#    #d2Udpsi2 = UnivariateSpline(psi_new[2:-2],diff_central(psi_new[1:-1],diff_central(psi_new,dl_B_new)),k=3,s=0,ext="const")
#    term1 = lambda x: gamma*x/U(x)*(dUdpsi(x))**2
#    term2 = lambda x: dUdpsi(x) + x*d2Udpsi2(x)
#    F = lambda x: term1(x) - term2(x)
#    if plotit:
#        z0_idx = np.abs(Z[:,0]).argmin()
#        end_idx = np.abs(Rho[z0_idx,:]-max_rho).argmin()
#        R_of_psi = UnivariateSpline(psi[z0_idx,0:end_idx],Rho[z0_idx,0:end_idx],s=0)
#        psi_test = np.linspace(psi_min,psi_edge,1000)
#        psi_norm = psi_test/psi_edge
#        fig9,(ax9a,ax9b,ax9c) = plt.subplots(3,1,sharex="col")
#        ax9a.plot(psi_new,dl_B_new,"o")
#        ax9a.plot(psi_test,U(psi_test),"r")
#        ax9b.plot(psi_new[1:-1],dUdpsi(psi_new[1:-1]),"o")
#        ax9b.plot(psi_test,dUdpsi(psi_test),"r")
#        ax9c.plot(psi_new[2:-2],d2Udpsi2(psi_new[2:-2]),"o")
#        ax9c.plot(psi_test,d2Udpsi2(psi_test),"r")
#        fig0,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2,sharex="all",figsize=(18,9))
#        ax0.plot(psi_new/psi_edge,dl_B_new,"o")
#        ax0.plot(psi_norm,U(psi_test),lw=2)
#        ax0.set_ylabel("U")
#        ax0top = ax0.twiny()
#        new_labels = ["{0:1.2f}".format(r) for r in R_of_psi(psi_edge*np.array(ax0.get_xticks()))]
#        ax0top.set_xticklabels(new_labels)
#        ax0top.set_xlabel("R (m)")
#        ax1.plot(psi_norm,dUdpsi(psi_test),'o')
#        ax1.set_ylabel("U'")
#        ax1top = ax1.twiny()
#        ax1top.set_xticklabels(new_labels)
#        ax1top.set_xlabel("R (m)")
#        ax2.plot(psi_norm,term1(psi_test),'o')
#        ax2.plot(psi_norm,term2(psi_test),'o')
#        ax2.set_xlabel("$\\psi/\\psi_{lim}$")
#        ax2.set_ylabel("Term1 and term2")
#        F_clean = dh.smooth(F(psi_test),20)
#        ax3.plot(psi_norm,F_clean,lw=2)
#        ax3.set_xlabel("$\\psi/\\psi_{lim}$")
#        ax3.set_ylabel("F($\\psi$)")
#        #ax3.set_ylim(-1.2,1.2)
#        plt.tight_layout()
#    return F

# Added by Roger some simple check for field lines looping back on them selves
# def get_F_v2(Rho,Z,psi,B,min_rho,max_rho,start_coord=None,end_coord=None,clockwise=True,plotit=False,plotdots=True,thresh=.2,num_psi=500):
#     gamma = 5.0/3.0
#     figf,axf = plt.subplots()
#     psi_min,psi_edge = locs_to_vals(Rho,Z,psi,[(min_rho,0),(max_rho,0)])
#     psi_levels = np.linspace(psi_min,psi_edge,num_psi)
#     cff = axf.contour(Rho,Z,psi,tuple(psi_levels))
#     dl_B_list = []
#     for psi0 in psi_levels:
#         if psi0 == 0.0:
#             flpoints = get_fieldlines(cff,psi0,start_coord=start_coord,end_coord=end_coord,clockwise=False)
#         else:
#             flpoints = get_fieldlines(cff,psi0,start_coord=start_coord,end_coord=end_coord,clockwise=clockwise)
#             if np.abs(flpoints[0][0]-start_coord[0]) > thresh:
#                 raise ValueError("I think some of these contours start after the separatrix.")
#         if plotdots:
#             x,y = flpoints[:,0],flpoints[:,1]
#             axf.plot(x,y,'bo')
#             axf.plot(x[0],y[0],'go')
#             axf.plot(x[-1],y[-1],'ro')
#         else:
#             plt.close()
#         dl_B_list.append(flux_surface_avg(Rho,Z,B,flpoints))
#     psi_new = psi_levels
#     dl_B_new = np.array(dl_B_list)
#     dl_B_new = dh.smooth(dl_B_new,5,mode="valid")
#     psi_new = dh.smooth(psi_new,5,mode="valid")
#     U = UnivariateSpline(psi_new,dl_B_new,k=4,s=0)
#     dUdpsi = UnivariateSpline(psi_new[1:-1],diff_central(psi_new,dl_B_new),k=1,s=0,ext="const")
#     d2Udpsi2 = UnivariateSpline(psi_new[2:-2],diff_central(psi_new[1:-1],diff_central(psi_new,dl_B_new)),k=1,s=0,ext="const")
#     #dUdpsi = UnivariateSpline(psi_new[1:-1],diff_central(psi_new,dl_B_new),k=3,s=0,ext="const")
#     #d2Udpsi2 = UnivariateSpline(psi_new[2:-2],diff_central(psi_new[1:-1],diff_central(psi_new,dl_B_new)),k=3,s=0,ext="const")
#     term1 = lambda x: gamma*x/U(x)*(dUdpsi(x))**2
#     term2 = lambda x: dUdpsi(x) + x*d2Udpsi2(x)
#     F = lambda x: term1(x) - term2(x)
#     if plotit:
#         z0_idx = np.abs(Z[:,0]).argmin()
#         end_idx = np.abs(Rho[z0_idx,:]-max_rho).argmin()
#         R_of_psi = UnivariateSpline(psi[z0_idx,0:end_idx],Rho[z0_idx,0:end_idx],s=0)
#         psi_test = np.linspace(psi_min,psi_edge,1000)
#         psi_norm = psi_test/psi_edge
#         fig9,(ax9a,ax9b,ax9c) = plt.subplots(3,1,sharex="col")
#         ax9a.plot(psi_new,dl_B_new,"o")
#         ax9a.plot(psi_test,U(psi_test),"r")
#         ax9b.plot(psi_new[1:-1],dUdpsi(psi_new[1:-1]),"o")
#         ax9b.plot(psi_test,dUdpsi(psi_test),"r")
#         ax9c.plot(psi_new[2:-2],d2Udpsi2(psi_new[2:-2]),"o")
#         ax9c.plot(psi_test,d2Udpsi2(psi_test),"r")
#         fig0,((ax0,ax1),(ax2,ax3)) = plt.subplots(2,2,sharex="all",figsize=(18,9))
#         ax0.plot(psi_new/psi_edge,dl_B_new,"o")
#         ax0.plot(psi_norm,U(psi_test),lw=2)
#         ax0.set_ylabel("U")
#         ax0top = ax0.twiny()
#         new_labels = ["{0:1.2f}".format(r) for r in R_of_psi(psi_edge*np.array(ax0.get_xticks()))]
#         ax0top.set_xticklabels(new_labels)
#         ax0top.set_xlabel("R (m)")
#         ax1.plot(psi_norm,dUdpsi(psi_test),'o')
#         ax1.set_ylabel("U'")
#         ax1top = ax1.twiny()
#         ax1top.set_xticklabels(new_labels)
#         ax1top.set_xlabel("R (m)")
#         ax2.plot(psi_norm,term1(psi_test),'o')
#         ax2.plot(psi_norm,term2(psi_test),'o')
#         ax2.set_xlabel("$\\psi/\\psi_{lim}$")
#         ax2.set_ylabel("Term1 and term2")
#         F_clean = dh.smooth(F(psi_test),20)
#         ax3.plot(psi_norm,F_clean,lw=2)
#         ax3.set_xlabel("$\\psi/\\psi_{lim}$")
#         ax3.set_ylabel("F($\\psi$)")
#         #ax3.set_ylim(-1.2,1.2)
#         plt.tight_layout()
#     return F

