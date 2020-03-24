#####  still need to remove datahelpers dependency!
#
#import analysis.datahelpers as dh
####import plottingtools.plottingtools as ptools
#from pleiades.helpers import get_fieldline_distance, get_fieldlines, interp
#import numpy as np
#import matplotlib.pyplot as plt
#from scipy.interpolate import InterpolatedUnivariateSpline, UnivariateSpline
#from scipy.linalg import eig
#
#
#
#
#
#"""
#Creates a tridiagonal matrix from the input arrays.
#
#Arguments:
#    a :                 the vector of values for the lower diagonal
#                        length (N-1)
#    b :                 the vector for the matrix diagonal
#                        length (N)
#    c :                 the vector for the upper diagonal
#                        length (N-1)
#
#Returns:
#    A tridiagonal matrix created from the arrays a, b, c
#
#Notes:
#
#"""
#def tridiag(a, b, c, k1=-1, k2=0, k3=1):
#    return np.diag(a, k1) + np.diag(b, k2) + np.diag(c, k3)
#
#
#
#
#
#"""
#Returns the curvature of a field line defined by the (r,z) points in flpoints
#
#Arguments:
#    flpoints :          array of [r,z] points that describe the field line
#    plotit :            whether or not to plot stuff
#
#Returns:
#    Returns an array containing the curvature of the field line.
#
#Notes:
#
#"""
#def curvature(flpoints, plotit = False):
#
#    x,y = flpoints[:,0],flpoints[:,1]
#    s = get_fieldline_distance(flpoints)
#
#    fx = InterpolatedUnivariateSpline(s,x,ext="zeros")
#    fy = InterpolatedUnivariateSpline(s,y,ext="zeros")
#
#    s_lin = np.linspace(np.min(s),np.max(s),1000)
#
#    fx_new = InterpolatedUnivariateSpline(s_lin,dh.smooth(fx(s_lin),20),ext="zeros")
#    fy_new = InterpolatedUnivariateSpline(s_lin,dh.smooth(fy(s_lin),20),ext="zeros")
#
#    x_prime = dh.smooth(fx_new.derivative(1)(s_lin),20)
#    x_dprime = dh.smooth(fx_new.derivative(2)(s_lin),20)
#    y_prime = dh.smooth(fy_new.derivative(1)(s_lin),20)
#    y_dprime = dh.smooth(fy_new.derivative(2)(s_lin),20)
#
#    curvature = (x_prime*y_dprime - y_prime*x_dprime) / np.power(x_prime**2 + y_prime**2, 3 / 2)
#    f_curv = InterpolatedUnivariateSpline(s_lin,dh.smooth(curvature,20),ext="zeros")
#
#    if plotit:
#        fig, ax = plt.subplots(2, 3, sharex = True, figsize=(14,7))
#
#        ax[0,0].plot(s,x,'bo')
#        ax[0,0].plot(s_lin,dh.smooth(fx(s_lin),20),'r')
#        ax[0,0].plot(s_lin,fx_new(s_lin),'g-')
#        ax[0,0].legend(['x data','smoothed spline 1','spline 2'])
#
#        ax[0,1].plot(s,y,'bo')
#        ax[0,1].plot(s_lin,dh.smooth(fy(s_lin),20),'r')
#        ax[0,1].plot(s_lin,fy_new(s_lin),'g-')
#        ax[0,1].legend(['y data','smoothed spline 1','spline 2'])
#
#        ax[1,0].plot(s_lin,x_prime,'bo')
#        ax[1,0].plot(s_lin,x_dprime,'ro')
#        ax[1,0].set_xlabel('s')
#        ax[1,0].legend(["x'","x''"])
#
#        ax[1,1].plot(s_lin,y_prime,'bo')
#        ax[1,1].plot(s_lin,y_dprime,'ro')
#        ax[1,1].set_xlabel('s')
#        ax[1,1].legend(["y'","y''"])
#
#        ax[0,2].plot(s_lin,curvature,'bo')
#        ax[0,2].plot(s_lin,f_curv(s_lin),'r')
#        ax[0,2].legend(["curvature","curvature spline"])
#
#        ax[1,2].set_xlabel('s')
#
#        fig.tight_layout()
#        #ptools.makePlot(fig,ax)
#        plt.show()
#
#    return -1*f_curv(s)
#
#
#
#
#
#"""
#Returns the parameters necessary for computing the high m stability.
#
#Arguments:
#    R,Z :               the 2D R and Z grid
#    B,psi_eq :          equilibrium magnetic field and Psi on the 2D gird
#    Pfunc :             the equilibrium pressure function at the midplane
#
#    psi_values :        the psi values where the parameters should be returned
#
#    start_coord :       (r,z) where the parameters along the field line should start
#                        the closest point on the field line to this point will be the start
#                        z should be positive
#    end_coord :         the position where the parameters should stop
#
#    num_points :        number of evenly spaced points to use when returning parameters
#                        along the field line
#    plotit :            if True plots stuff
#
#Returns:
#    Returns an array of parameters of length psi_values.
#    Each index in the array is an array with the following:
#    [flpoints_new,b_fl,c_fl,d_fl,p_fl,dp_dpsi_fl] where flpoints_new are the [r,z]
#    evenly spaced field line points, b_fl is the magentic field along the field line,
#    c_fl is the curvature, d_fl is the density, p_fl is the pressure and dp_dpsi_fl is
#    the pressure gradient.
#
#Notes:
#
#"""
#def get_params(R, Z, B, psi_eq, Pfunc, psi_values, start_coord, end_coord, num_points = 201, plotit = False):
#
#    psi = psi_eq.flatten()
#    r = R.flatten()
#    z = Z.flatten()
#    z0_idx = np.where(z==0.0)
#    r_z0 = r[z0_idx]
#    p_z0 = np.array(map(Pfunc,r_z0))
#    psi_z0 = psi[z0_idx]
#    lim_idx = (i for i,p in enumerate(p_z0) if p == 0.0).next()
#    p_psifit = InterpolatedUnivariateSpline(psi_z0[0:lim_idx+1],p_z0[0:lim_idx+1],ext="zeros")
#    pprime_fit = p_psifit.derivative()
#    psi_lin = np.linspace(psi_z0.min(),psi_z0.max(),1000)
#
#    ### PLOT PRESSURE and PRESSURE GRADIENT
#    if plotit:
#        fig,(ax1,ax2) = plt.subplots(2,1,sharex="col",figsize=(10,7))
#        ax1.plot(psi_z0,pprime_fit(psi_z0),"bo")
#        ax1.plot(psi_lin,pprime_fit(psi_lin),"r",lw=2)
#        [ax1.axvline(_x, linewidth=1, ls='--', color='k') for _x in psi_values]
#        ax1.set_ylabel('Pressure Gradient')
#        ax2.plot(psi_z0,p_psifit(psi_z0),"bo")
#        ax2.plot(psi_lin,p_psifit(psi_lin),"r",lw=2)
#        [ax2.axvline(_x, linewidth=1, ls='--', color='k') for _x in psi_values]
#        ax2.set_ylabel('Pressure')
#        ax2.set_xlabel('$\psi$')
#        #ptools.makePlot(fig,[ax1,ax2])#,filename='pressure',fileformat='eps')
#        plt.show()
#
#    params_array = []
#
#    for psi_test in psi_values:
#        p_fl = p_z0[np.argmin(np.abs(psi_z0-psi_test))]
#        dp_dpsi_fl = pprime_fit(psi_z0[np.argmin(np.abs(psi_z0-psi_test))])
#
#        k = 1.38*10**-23
#        T = 10 * (1.6*10**-19)/k #10ev in SI
#        m_ion = 1.67*10**-27
#        d_fl = p_fl/(k*T)*m_ion
#
#        ### GET FIELD LINE POINTS
#        fig = plt.figure()
#        ax = plt.gca()
#        cf1 = ax.contour(R,Z,np.abs(psi_eq),[psi_test],colors='r',zorder=2)
#        plt.close()
#        flpoints = get_fieldlines(cf1,psi_test,start_coord=start_coord,end_coord=end_coord,clockwise=False)
#
#        ### INTERPOLATE ONTO EVENLY SPACED GRID
#        r_fl,z_fl = flpoints[:,0],flpoints[:,1]
#        z = np.array(list(z_fl)+[-z_fl[0]])
#        z = z[::-1] #flip
#        r = np.array(list(r_fl)+[r_fl[0]])
#        r = r[::-1]
#        flpoints = np.vstack((z,r)).T
#        fl_dist = get_fieldline_distance(flpoints)
#        spl = UnivariateSpline(z,r,k=1,s=0)
#        fl_spl = UnivariateSpline(fl_dist,z,k=1,s=0)
#        uniform_s = np.linspace(fl_dist[0],fl_dist[-1],num_points)  ###this sets num points for flpoints
#        zlimit = fl_spl(uniform_s)
#        rlimit = spl(zlimit)
#
#        ### PLOT EVENLY SPACED GRID
#        if plotit:
#            fig = plt.figure()
#            ax = plt.gca()
#            ax.set_aspect('equal')
#            ax.plot(z,r,"bo")
#            ax.plot(zlimit,rlimit,"ro")
#            ax.set_xlabel('z (m)')
#            ax.set_ylabel('r (m)')
#            ax.legend(['original','evenly spaced'])
#            #ptools.makePlot(fig,[ax])
#            plt.show()
#
#        ### WORK WITH NEW EVENLY SPACED GRID
#        flpoints_new = np.vstack((rlimit,zlimit)).T
#
#        c_fl = curvature(flpoints_new, plotit = False)
#        b_fl = interp(R, Z, B, flpoints_new)
#
#        if plotit:
#            s = get_fieldline_distance(flpoints_new)
#            fig = plt.figure()
#            ax = plt.gca()
#            ax.plot(s,c_fl,'ko')
#            ax.plot(s,b_fl,'bo')
#            ax.plot(s,rlimit,'ro')
#            ax.set_xlabel('s (m)')
#            ax.legend(['curvature','mod B','radius'])
#            #ptools.makePlot(fig,[ax])
#            plt.show()
#
#        params = [flpoints_new,b_fl,c_fl,d_fl,p_fl,dp_dpsi_fl]
#        params_array.append(params)
#
#    return params_array
#
#
#
#
#
#"""
#Uses the generalized eigenvalue matrix method to solve for the eigenvalues and
#eigenvectors of the coupled system given in equations (14) and (15) of the
#paper by K. V. Lotov
#
#Arguments:
#    params :            vector of the parameters along field line:
#                            [flpoints,B,curvature,density,Pressure,dp/dpsi]
#    bc_z_limit :        a positive value indicating the boundary of the desired field line in z
#                        arrays will be clipped outside the region [-z,z]
#    plotit :            whether or not to plot stuff
#
#Returns:
#    Returns (vals,vecs,sx,sy) where vals is an array of the eigenvalues
#    sorted by magnitude, vecs is a matrix where column i corresponds to
#    the eigenvector associated with vals[i], sx is the array of s (arclength) values
#    where X was computed and sy is the same but for Y. sx and sy are a subset of the original
#    arclength defined by flpoints in params.
#
#Notes:
#
#"""
#def high_m_stability(params, bc_z_limit = None, plotit = False):
#
#    flpoints,By,ky,rho,press,dpdphi = params
#    ry,zy = flpoints[:,0],flpoints[:,1]
#
#    s = get_fieldline_distance(flpoints)
#    ds = s[-1]/(len(s)-1)
#
#    if bc_z_limit:
#        print('Clipping arrays to fit in boundary...')
#        f_half = zy[:len(zy)/2]
#        z_ind = np.argmin( np.abs(np.abs(f_half)-bc_z_limit) )
#
#        s = s[z_ind:-z_ind]
#        zy = zy[z_ind:-z_ind]
#        ry = ry[z_ind:-z_ind]
#        By = By[z_ind:-z_ind]
#        ky = ky[z_ind:-z_ind]
#
#    if plotit:
#        fig = plt.figure()
#        ax = plt.gca()
#        ax.plot(s,ky,'ko')
#        ax.plot(s,By,'bo')
#        ax.plot(s,ry,'ro')
#        ax.legend(['curvature','mod B','radius'])
#        ax.set_title('Arrays as a function of s')
#        #ptools.makePlot(fig,[ax])#,filename='arrays',fileformat='eps')
#        plt.show()
#
#    #may want to 'symmetrize' the arrays by taking the average of each s value
#    for i in range(len(s)/2):
#        ry[i] = ry[-i-1] = (ry[i]+ry[-i-1])/2.0
#        By[i] = By[-i-1] = (By[i]+By[-i-1])/2.0
#        ky[i] = ky[-i-1] = (ky[i]+ky[-i-1])/2.0
#    #rescale equations so numbers are all the same order
#
#    gamma = 5.0/3.0
#    mu0 = 4*np.pi*10**(-7)
#    N = len(s)-1 #number of segments
#    NX = N
#    NY = N-1
#
#    # compute B, r and kappa at X points
#    Bx = 0.5*(By[0:NX] + By[1:NX+1])
#    By = By[1:NX]  # DROP s=0 and N*ds, NY unknown interior Y nodes
#    rx = 0.5*(ry[0:NX] + ry[1:NX+1])
#    ry = ry[1:NX]  # DROP s=0 and N*ds, NY unknown interior Y nodes
#    kx = 0.5*(ky[0:NX] + ky[1:NX+1])
#    ky = ky[1:NX]  # DROP s=0 and N*ds, NY unknown interior Y nodes
#
#    '''
#    X equations
#    '''
#    a = np.zeros(NX)
#    b = np.zeros(NX)
#    c = np.zeros(NX)
#
#    a[1:NX]  = 1./By[0:NX-1]/ds**2
#    c[0:NX-1]= 1./By[0:NX-1]/ds**2
#    b[1:NX-1]= -a[1:NX-1] -c[1:NX-1]
#    b[0] = -c[0]
#    b[-1] = -a[-1]
#    Mxx = tridiag(a[1:NX],b,c[0:NX-1])
#
#    # right hand side matrix [Nxx, Nxy]
#    Nxx = -np.diag((rho/(gamma*press*Bx) + mu0*rho/Bx**3),0)
#
#    b = -rho*kx/(Bx**2*rx)
#    Nxy = np.diag(b,0) + np.diag(b[1:NX],-1)
#    Nxy = Nxy[0:NX,0:NY]
#
#    '''
#    Y equations
#    '''
#    a = np.zeros(NY)
#    b = np.zeros(NY)
#    c = np.zeros(NY)
#
#    a[0:NY] = 1./(Bx[0:NY]*rx[0:NY]**2)/(ds**2)
#    c[0:NY] = 1./(Bx[1:NY+1]*rx[1:NY+1]**2)/(ds**2)
#    b[1:NY-1] = -a[1:NY-1] -c[1:NY-1]
#    b[0] = -a[0] -c[0]
#    b[-1] = -a[-1] -c[-1]
#
#    d = mu0*ky/(By**2*ry)
#    b = b+2*d*dpdphi
#    Myy = tridiag(a[1:NY],b,c[0:NY-1])
#
#    Myx = np.diag(d,1) + np.diag( np.concatenate((d,[d[-1]])) ,0) #this last value gets cropped in next line
#    Myx = Myx[0:NY,0:NX]
#
#    Nyy = -np.diag(mu0*rho/((ry*By)**2*By))
#
#    '''
#    Full matrix
#    '''
#    M = np.concatenate((np.concatenate((Mxx, np.zeros((NX,NY))), axis=1), np.concatenate((Myx, Myy), axis=1)), axis=0)
#    N = np.concatenate((np.concatenate((Nxx, Nxy), axis=1), np.concatenate((np.zeros((NY,NX)), Nyy), axis=1)), axis=0)
#
#
#    vals, vecs = eig(M,b=N)
#    #print(vals)
#    sort_index = np.argsort(np.abs(vals))
#
#    vals = vals[sort_index]
#    vecs = vecs[:,sort_index]
#
#    sx = np.linspace(s[0]+ds/2,s[-1]-ds/2,NX)
#    sy = np.linspace(s[1],s[-2],NY)
#
#    if plotit:
#        omega_str = ''
#
#        sx_cent = np.linspace(-s[-1]/2,s[-1]/2,NX) #maybe not actually exact points (probs want sx-arclength/2)
#        sy_cent = np.linspace(-s[-1]/2,s[-1]/2,NY) #maybe not actually exact points
#
#        fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(13,7))
#        lis = np.arange(0,5)
#        for i in lis:
#            ax1.plot(sx_cent,vecs[:NX,i],linewidth=2)
#            ax2.plot(sy_cent,vecs[NX:,i],linewidth=2)
#            omega_str += ' $\omega_{'+str(i)+'}^2$=%.4e ' % vals[i].real
#
#        ax1.axvline(x=0.0,ls='--',color='k',linewidth=1)
#        ax1.set_title('sorted magnitude eigenvalues\n'+omega_str,fontsize=18)
#        ax1.legend([str(x) for x in lis])
#        ax1.set_ylabel('X')
#
#        ax2.axvline(x=0.0,ls='--',color='k',linewidth=1)
#        ax2.legend([str(x) for x in lis])
#        ax2.set_xlabel('s (centered at z=0)')
#        ax2.set_ylabel('Y')
#
#        #ptools.makePlot(fig,[ax1,ax2])#,filename='highermodes',fileformat='eps')
#        plt.show()
#
#    return vals,vecs,sx,sy
#
#
#
#
#
#"""
#Uses 4th order rugge kutta method to solve the set of equations for an IVP.
#
#Arguments:
#    w0 :                vector of the initial state variables:
#                            w = [X0,Y0,F0,G0]
#                        F is the derivative of X and G is the derivative of Y
#    params :            vector of the parameters along field line:
#                            [flpoints,B,curvature,density,Pressure,dp/dpsi]
#    omega:              omega^2 in equations
#    num_points :        number of points to use
#    bc_z_limit :        a positive value indicating the boundary of the desired field line in z
#                        arrays will be clipped outside the region [-z,z]
#    start_middle :      if True the initial conditions will be applied in the center of the field line
#    plotit :            whether or not to plot stuff
#
#Returns:
#    Returns (X,Y,F,G,s)
#
#Notes:
#
#"""
#def rk4(w0, params, omega, num_points = 250, bc_z_limit = None, start_middle = False, plotit = False):
#
#    flpoints,By,ky,rho,press,dpdphi = params
#    ry,zy = flpoints[:,0],flpoints[:,1]
#
#    s = get_fieldline_distance(flpoints)
#    fr = InterpolatedUnivariateSpline(s,ry,ext="zeros")
#    fB = InterpolatedUnivariateSpline(s,By,ext="zeros")
#    fk = InterpolatedUnivariateSpline(s,ky,ext="zeros")
#    fdB = fB.derivative()
#    fdr = fr.derivative()
#
#    if bc_z_limit:
#        print('Clipping arrays to fit in boundary...')
#        f_half = zy[:len(zy)/2]
#        z_ind = np.argmin( np.abs(np.abs(f_half)-bc_z_limit) )
#
#        s = s[z_ind:-z_ind]
#
#    if start_middle:
#        s = s[len(s)/2:]
#
#    s = np.linspace(s[0],s[-1],num_points)
#
#    if plotit:
#        fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(13,7))
#        ax1.plot(s,fr(s),linewidth=2)
#        ax1.plot(s,fB(s),linewidth=2)
#        ax1.plot(s,fk(s),linewidth=2)
#        ax1.legend(['radius','magnetic field','curvature'])
#        ax1.set_title('Parameters as a function of s over the solving range')
#        ax2.plot(s,fdr(s),linewidth=2)
#        ax2.plot(s,fdB(s),linewidth=2)
#        ax2.legend(['radius derivative','B derivative'])
#        ax2.set_xlabel('s (m)')
#        #ptools.makePlot(fig,[ax1,ax2])
#        plt.show()
#
#    p = [fr, fB, fk, rho, press, dpdphi, fdB, fdr]
#
#    X0, Y0, F0, G0 = w0
#
#    vx = [0] * len(s)
#    vy = [0] * len(s)
#    vf = [0] * len(s)
#    vg = [0] * len(s)
#
#    vx[0] = X = X0
#    vy[0] = Y = Y0
#    vf[0] = F = F0
#    vg[0] = G = G0
#
#    for i in range(1, len(s)):
#        #print(i)
#        h = s[i] - s[i-1]
#
#        k0 = h * __x(s[i-1], X, Y, F, G, p, omega)
#        l0 = h * __y(s[i-1], X, Y, F, G, p, omega)
#        m0 = h * __f(s[i-1], X, Y, F, G, p, omega)
#        n0 = h * __g(s[i-1], X, Y, F, G, p, omega)
#
#        k1 = h * __x(s[i-1]+0.5*h, X+0.5*k0, Y+0.5*l0, F+0.5*m0, G+0.5*n0, p, omega)
#        l1 = h * __y(s[i-1]+0.5*h, X+0.5*k0, Y+0.5*l0, F+0.5*m0, G+0.5*n0, p, omega)
#        m1 = h * __f(s[i-1]+0.5*h, X+0.5*k0, Y+0.5*l0, F+0.5*m0, G+0.5*n0, p, omega)
#        n1 = h * __g(s[i-1]+0.5*h, X+0.5*k0, Y+0.5*l0, F+0.5*m0, G+0.5*n0, p, omega)
#
#        k2 = h * __x(s[i-1]+0.5*h, X+0.5*k1, Y+0.5*l1, F+0.5*m1, G+0.5*n1, p, omega)
#        l2 = h * __y(s[i-1]+0.5*h, X+0.5*k1, Y+0.5*l1, F+0.5*m1, G+0.5*n1, p, omega)
#        m2 = h * __f(s[i-1]+0.5*h, X+0.5*k1, Y+0.5*l1, F+0.5*m1, G+0.5*n1, p, omega)
#        n2 = h * __g(s[i-1]+0.5*h, X+0.5*k1, Y+0.5*l1, F+0.5*m1, G+0.5*n1, p, omega)
#
#        k3 = h * __x(s[i-1]+h, X+k2, Y+l2, F+m2, G+n2, p, omega)
#        l3 = h * __y(s[i-1]+h, X+k2, Y+l2, F+m2, G+n2, p, omega)
#        m3 = h * __f(s[i-1]+h, X+k2, Y+l2, F+m2, G+n2, p, omega)
#        n3 = h * __g(s[i-1]+h, X+k2, Y+l2, F+m2, G+n2, p, omega)
#
#        vx[i] = X = X + (k0+k1+k1+k2+k2+k3)/6.0
#        vy[i] = Y = Y + (l0+l1+l1+l2+l2+l3)/6.0
#        vf[i] = F = F + (m0+m1+m1+m2+m2+m3)/6.0
#        vg[i] = G = G + (n0+n1+n1+n2+n2+n3)/6.0
#
#        #print(X,Y,F,G)
#
#    if plotit:
#        #s_centered = np.linspace(-s[-1]/2,s[-1]/2,len(s))
#        fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(13,7))
#        ax1.plot(s,vx,linewidth=2)
#        ax2.plot(s,vy,linewidth=2)
#
#        ax1.set_title('X and Y as a function of s\n$\omega^2=%.4e$' % omega)
#        ax1.set_ylabel('X')
#
#        ax2.set_xlabel('s')
#        ax2.set_ylabel('Y')
#
#        #ptools.makePlot(fig,[ax1,ax2])
#        plt.show()
#
#    return vx, vy, vf, vg, s
#
#
#
#def __x(s, X, Y, F, G, p, omega):
#    return F
#
#def __y(s, X, Y, F, G, p, omega):
#    return G
#
#def __f(s, X, Y, F, G, p, omega):
#
#    gamma = 5.0/3.0
#    u0 = 4*np.pi*10**(-7)
#
#    r, B, k, rho, press, dpdphi, dB, dr = p
#
#    term1 = -2*k(s)*omega/(B(s)*r(s))*Y
#    term2 = -omega*(1/(gamma*press) + u0/B(s)**2)*X
#    term3 = dB(s)/(rho*B(s))*F
#
#    return rho*(term1+term2+term3)
#
#def __g(s, X, Y, F, G, p, omega):
#
#    u0 = 4*np.pi*10**(-7)
#
#    r, B, k, rho, press, dpdphi, dB, dr = p
#
#    term1 = dB(s)/(u0*B(s)*r(s)**2)*G
#    term2 = 2*dr(s)/(u0*r(s)**3)*G
#    term3 = -omega*rho/(r(s)**2*B(s)**2)*Y
#    term4 = -2*k(s)*dpdphi/(B(s)*r(s))*Y
#    term5 = -2*k(s)/(B(s)*r(s))*X
#
#    return u0*r(s)**2*(term1+term2+term3+term4+term5)
