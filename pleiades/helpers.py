from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import scipy.integrate
from scipy.interpolate import interp1d, interp2d, RectBivariateSpline,UnivariateSpline,griddata
from scipy.spatial import Delaunay, ConvexHull
from matplotlib.collections import LineCollection
#import analysis.datahelpers as dh

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

def write_eqdsk(Rho,Z,psi,plas_currents,fname,title):
    title = str(title)+" cursign,nnr,nnz,nnv   = "
    nnr,nnz,nnv = int(len(Rho[0,:])),int(len(Z[:,0])),101
    rbox,zbox = np.amax(Rho)-np.amin(Rho),np.amax(Z)-np.amin(Z)
    tot_cur = np.sum(plas_currents)
    cursign = np.sign(tot_cur)
    blank = np.zeros(nnv)
    limit_pairs,vessel_pairs = 100,100
    rho0_idx = np.abs(Rho[0,:]).argmin()
    #### Plotting current and flux lines from plasma currents 
    [psi_lim] = locs_to_vals(Rho,Z,psi,[(.9,0)])
    psi_ves = psi_lim*1.02
    psi_levels = tuple(sorted([psi_lim,psi_ves]))
    fig,ax = plt.subplots()
    cf = ax.contour(Rho,Z,psi,psi_levels,colors='k',zorder=1)
    ## get contour for psi_lim boundary
    flpoints = get_fieldlines(cf,psi_lim,start_coord=(.002,.6),end_coord=(.002,-.6))
    r,z = flpoints[:,0],flpoints[:,1]
    z = np.array(list(z)+[-z[0]])
    z = z[::-1]
    r = np.array(list(r)+[r[0]])
    r = r[::-1]
    flpoints = np.vstack((z,r)).T
    fl_dist = get_fieldline_distance(flpoints)
    spl = UnivariateSpline(z,r,k=1,s=0)
    fl_spl = UnivariateSpline(fl_dist,z,k=1,s=0)
    uniform_s = np.linspace(fl_dist[0],fl_dist[-1],100)
    zlimit = fl_spl(uniform_s)
    rlimit = spl(zlimit)
    ax.plot(r,z,"bo")
    ax.plot(rlimit,zlimit,"ro")
    ## get contour for psi_ves boundary
    flpoints = get_fieldlines(cf,psi_ves,start_coord=(.002,.6),end_coord=(.002,-.6))
    r,z = flpoints[:,0],flpoints[:,1]
    z = np.array(list(z)+[-z[0]])
    z = z[::-1]
    r = np.array(list(r)+[r[0]])
    r = r[::-1]
    flpoints = np.vstack((z,r)).T
    fl_dist = get_fieldline_distance(flpoints)
    spl = UnivariateSpline(z,r,k=1,s=0)
    fl_spl = UnivariateSpline(fl_dist,z,k=1,s=0)
    uniform_s = np.linspace(fl_dist[0],fl_dist[-1],100)
    zves = fl_spl(uniform_s)
    rves = spl(zves)
    ax.plot(r,z,"yo")
    ax.plot(rves,zves,"go")
    plt.show()

    lim_ves_pairs = [loc for pair in zip(rlimit,zlimit) for loc in pair]+[loc for pair in zip(rves,zves) for loc in pair]

    with open(fname,"wb") as f:
        ## line 1 -- write grid information: cursign, nnr, nnz, nnv
        f.write(title+"".join("{:4d}".format(xi) for xi in [int(cursign),nnr,nnz,nnv]))
        ## line 2 -- write rbox,zbox,0,0,0
        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [rbox,zbox,0,0,0]))
        ## line 3 -- write 0,0,0,psi_lim,0
        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [0,0,0,psi_lim,0]))
        ## line 4 -- write total toroidal current,0,0,0,0
        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [tot_cur,0,0,0,0]))
        ## line 5 -- write 0,0,0,0,0
        f.write("\n"+"".join("{: 16.9E}".format(xi) for xi in [0,0,0,0,0]))
        ## line 6 -- write list of toroidal flux for each flux surface (zeros)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(blank)))
        ## line 7 -- write list of pressure for each flux surface (zeros)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(blank)))
        ## line 8 -- write list of (RBphi)' for each flux surface (zeros)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(blank)))
        ## line 9 -- write list of P' for each flux surface (zeros)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(blank)))
        ## line 10 -- write flattened list of psi values on whole grid (NOT ZERO)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(psi.flatten())))
        ## line 11 -- write list of q for each flux surface (zeros)
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(blank)))
        ## line 12 -- write number of coordinate pairs for limit surface and vessel surface
        f.write("\n"+"".join("{0:5d}{1:5d}".format(limit_pairs,vessel_pairs)))
        ## line 13 -- write list of R,Z pairs for limiter surface, then vessel surface
        f.write("\n"+"".join("{: 16.9E}\n".format(xi) if np.mod(i+1,5)==0 else "{: 16.9E}".format(xi) for i,xi in enumerate(lim_ves_pairs)))

def read_eqdsk(filename):
    """
    line 1 -- read grid information: title cursign, nnr, nnz, nnv
    line 2 -- read rbox,zbox,0,0,0
    line 3 -- read 0,0,0,psi_lim,0
    line 4 -- read total toroidal current,0,0,0,0
    line 5 -- read 0,0,0,0,0
    line 6 -- read list of toroidal flux for each flux surface (zeros)
    line 7 -- read list of pressure for each flux surface (zeros)
    line 8 -- read list of (RBphi)' for each flux surface (zeros)
    line 9 -- read list of P' for each flux surface (zeros)
    line 10 -- read flattened list of psi values on whole grid (NOT ZERO)
    line 11 -- read list of q for each flux surface (zeros)
    line 12 -- read number of coordinate pairs for limit surface and vessel surface
    line 13 -- read list of R,Z pairs for limiter surface, then vessel surface
    """
    eq_dict = {}
    with open(filename,"r") as f:
        lines = f.readlines()
        line1 = lines[0].split()
        eq_dict["cursign"] = int(line1[-4])
        eq_dict["nnr"] = int(line1[-3])
        eq_dict["nnz"] = int(line1[-2])
        eq_dict["nnv"] = int(line1[-1])
        line2 = lines[1].split()
        eq_dict["rbox"] = float(line2[-5])
        eq_dict["zbox"] = float(line2[-4])
        line3 = lines[2].split()
        eq_dict["psi_lim"] = float(line3[-2])
        line4 = lines[3].split()
        eq_dict["Ip"] = float(line4[-5])
        line5 = lines[4].split()
        fs_lines = int(np.ceil(eq_dict["nnv"]/5.0))
        head = [line.strip().split() for line in lines[5:5+fs_lines]]
        tor_flux = np.array([float(num) for line in head for num in line])
        eq_dict["tor_flux"] = tor_flux
        head = [line.strip().split() for line in lines[5+fs_lines:5+2*fs_lines]]
        p_flux = np.array([float(num) for line in head for num in line])
        eq_dict["p_flux"] = p_flux
        head = [line.strip().split() for line in lines[5+2*fs_lines:5+3*fs_lines]]
        rbphi_flux = np.array([float(num) for line in head for num in line])
        eq_dict["rbphi_flux"] = rbphi_flux
        head = [line.strip().split() for line in lines[5+3*fs_lines:5+4*fs_lines]]
        pprime_flux = np.array([float(num) for line in head for num in line])
        eq_dict["pprime_flux"] = pprime_flux
        # Read psi on whole grid, nnr x nnz
        nnr,nnz = eq_dict["nnr"],eq_dict["nnz"]
        rz_pts = nnr*nnz
        l0 = 5+4*fs_lines
        psi_lines = int(np.ceil(rz_pts/5.0))
        head = [line.strip().split() for line in lines[l0:l0+psi_lines]]
        psi = np.array([float(num) for line in head for num in line])
        eq_dict["psi"] = psi.reshape((nnr,nnz))
        rbox,zbox = eq_dict["rbox"],eq_dict["zbox"]
        R,Z = np.meshgrid(np.linspace(0,rbox,nnr),np.linspace(-zbox/2,zbox/2,nnz))
        eq_dict["R"] = R
        eq_dict["Z"] = Z
        head = [line.strip().split() for line in lines[l0+psi_lines:l0+psi_lines+fs_lines]]
        q_flux = np.array([float(num) for line in head for num in line])
        eq_dict["q_flux"] = q_flux
        nlim_pairs, nves_pairs = [int(x) for x in lines[l0+psi_lines+fs_lines].strip().split()]
        eq_dict["nlim_pairs"] = nlim_pairs
        eq_dict["nves_pairs"] = nves_pairs
        pair_lines = int(np.ceil((nlim_pairs + nves_pairs)*2.0/5.0))
        rest = [line.rstrip() for line in lines[l0+psi_lines+fs_lines+1:]]
        rest = [line[i:i+16] for line in rest for i in range(0,len(line),16)]
        pairs = np.array([float(num.strip()) for num in rest])
        lim_pairs = np.array(zip(pairs[0:2*nlim_pairs:2],pairs[1:2*nlim_pairs:2]))
        ves_pairs = np.array(zip(pairs[2*nlim_pairs::2],pairs[2*nlim_pairs+1::2]))
        eq_dict["lim_pairs"] = lim_pairs
        eq_dict["ves_pairs"] = ves_pairs
        print(lim_pairs.shape)
        print(ves_pairs.shape)

        


        return eq_dict
        
        

