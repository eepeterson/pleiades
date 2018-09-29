from __future__ import print_function, division, absolute_import, unicode_literals
from numpy import pi,linspace,meshgrid,sin,cos,sqrt,sum,array,ones,zeros,hstack,vstack,sign,mod,isfinite,ceil,isclose
from scipy.special import ellipk, ellipe
from multiprocessing import Process, Queue, cpu_count
#import tables
import warnings

def diff_12_central(x, y):
    x0 = x[:-2]
    x1 = x[1:-1]
    x2 = x[2:]
    y0 = y[:-2]
    y1 = y[1:-1]
    y2 = y[2:]
    f = (x2 - x1) / (x2 - x0)

    d_one = (1 - f) * (y2 - y1) / (x2 - x1) + f * (y1 - y0) / (x1 - x0)  # first derivative at x1 (dy/dx)
    d_two = 2 * (y0 / ((x1 - x0) * (x2 - x0)) + y1 / ((x1 - x0) * (x1 - x2)) + y2 / (
                (x2 - x0) * (x2 - x1)))  # second derivative at x1 (d^2y/dx^2)

    return d_one, d_two

def new_greens_test(R,Z):
    m,n = len(R),len(R)
    gpsi = zeros((m,n))
    R2 = R**2
    mu_0 = 4*pi*10**-7
    pre_factor = mu_0/(4*pi)
    for i,(r0,z0) in enumerate(zip(R,Z)):
        if isclose(r0,0,rtol=0,atol=1E-12):
            continue
        fac0 = (Z-z0)**2
        d = sqrt(fac0 + (R+r0)**2)
        d_ = sqrt(fac0 + (R-r0)**2)
        k_2 = 4*R*r0/d**2
        K = ellipk(k_2)
        E = ellipe(k_2)
        denom = d_**2*d
        fac1 = d_**2*K
        fac2 = (fac0 + R2 + r0**2)*E
        gpsi_tmp = pre_factor*R*r0/d * 4/k_2*((2-k_2)*K - 2*E)
        gpsi_tmp[~isfinite(gpsi_tmp)]=0
        gpsi[:,i] = gpsi_tmp
    return gpsi

def short_greens_test(R,Z):
    # must pass in 2D R and Z
    n,m = R.shape
    r,z = R.flatten(),Z.flatten()
    gpsi = zeros((m*n,m))
    r2 = r**2
    mu_0 = 4*pi*10**-7
    pre_factor = mu_0/(4*pi)
    for i,(r0,z0) in enumerate(zip(r[0:m],z[0:m])):
        if isclose(r0,0,rtol=0,atol=1E-12):
            continue
        fac0 = (z-z0)**2
        d = sqrt(fac0 + (r+r0)**2)
        d_ = sqrt(fac0 + (r-r0)**2)
        k_2 = 4*r*r0/d**2
        K = ellipk(k_2)
        E = ellipe(k_2)
        denom = d_**2*d
        fac1 = d_**2*K
        fac2 = (fac0 + r2 + r0**2)*E
        gpsi_tmp = pre_factor*r*r0/d * 4/k_2*((2-k_2)*K - 2*E)
        gpsi_tmp[~isfinite(gpsi_tmp)]=0
        gpsi[:,i] = gpsi_tmp
    return gpsi

def get_gpsi(R,Z):
    # must pass in 2D R and Z
    n,m = R.shape
    r,z = R.flatten(),Z.flatten()
    gpsis = zeros((m*n,m))
    gpsi = zeros((m*n,m*n))
    r2 = r**2
    mu_0 = 4*pi*10**-7
    pre_factor = mu_0/(4*pi)
    print("computing gpsi blocks...")
    for i,(r0,z0) in enumerate(zip(r[0:m],z[0:m])):
        if isclose(r0,0,rtol=0,atol=1E-12):
            continue
        fac0 = (z-z0)**2
        d = sqrt(fac0 + (r+r0)**2)
        d_ = sqrt(fac0 + (r-r0)**2)
        k_2 = 4*r*r0/d**2
        K = ellipk(k_2)
        E = ellipe(k_2)
        denom = d_**2*d
        fac1 = d_**2*K
        fac2 = (fac0 + r2 + r0**2)*E
        gpsi_temp = pre_factor*r*r0/d * 4/k_2*((2-k_2)*K - 2*E)
        gpsi_temp[~isfinite(gpsi_temp)]=0
        gpsis[:,i] = gpsi_temp
    print("creating reflected block matrix")
    gpsis2 = zeros((m*(2*n-1),m))
    gpsis2[(n-1)*m:,:] = gpsis
    for k in range(n-1):
        if k == 0:
            gpsis2[0:m,:] = gpsis[-m:,:]
        else:
            gpsis2[k*m:(k+1)*m,:] = gpsis[-(k+1)*m:-k*m,:]
    print("building huge matrix...")
    for p in range(n):
        gpsi[:,p*m:(p+1)*m] = gpsis2[(n-(p+1))*m:(2*n-(p+1))*m,:]
    print("returning...")
    return gpsi

def get_greens(R,Z,rzdir,out_q=None,out_idx=None):
    warnings.simplefilter("ignore",RuntimeWarning)
    m,n = len(R),len(rzdir)
    print(m,n)
    gpsi = zeros((m,n))
    gBR = zeros((m,n))
    gBZ = zeros((m,n))
    R2 = R**2
    mu_0 = 4*pi*10**-7
    pre_factor = mu_0/(4*pi)
    for i,(r0,z0,csign) in enumerate(rzdir):
        if isclose(r0,0,rtol=0,atol=1E-12):
            continue
        fac0 = (Z-z0)**2
        d = sqrt(fac0 + (R+r0)**2)
        d_ = sqrt(fac0 + (R-r0)**2)
        k_2 = 4*R*r0/d**2
        K = ellipk(k_2)
        E = ellipe(k_2)
        denom = d_**2*d
        fac1 = d_**2*K
        fac2 = (fac0 + R2 + r0**2)*E
        gpsi_tmp = csign*pre_factor*R*r0/d * 4/k_2*((2-k_2)*K - 2*E)
        gpsi_tmp[~isfinite(gpsi_tmp)]=0
        gpsi[:,i] = gpsi_tmp
        gBR_tmp = -2*csign*pre_factor*(Z-z0)*(fac1 - fac2)/(R*denom)
        gBR_tmp[~isfinite(gBR_tmp)]=0
        gBR[:,i] = gBR_tmp
        gBZ_tmp = 2*csign*pre_factor*(fac1 - (fac2-2*r0**2*E))/denom
        gBZ_tmp[~isfinite(gBZ_tmp)]=0
        gBZ[:,i] = gBZ_tmp
    out_tup = (gpsi,gBR,gBZ)
    if out_q is None:
        return out_tup
    else:
        if out_idx is None:
            raise ValueError("I don't know where to put this output, please specify out_idx")
        out_q.put((out_idx,)+out_tup)

def compute_greens(R,Z,rzdir=None,nprocs=1):
    warnings.simplefilter("ignore",RuntimeWarning)
    proc_max = cpu_count()
    if rzdir is None:
        rzdir = vstack((R,Z,ones(len(R)))).T
    m,n = len(R),len(rzdir)
    print(m, n)
    gpsi = zeros((m,n))
    gBR = zeros((m,n))
    gBZ = zeros((m,n))
    if nprocs > proc_max:
        nprocs = proc_max
    procs = []
    out_q = Queue()
    chunksize = int(ceil(rzdir.shape[0]/float(nprocs)))
    print(chunksize)
    for i in xrange(nprocs):
        p = Process(target=get_greens,args=(R,Z,rzdir[i*chunksize:(i+1)*chunksize,:]),kwargs={"out_q":out_q,"out_idx":i})
        procs.append(p)
        p.start()

    for j in xrange(nprocs):
        print("getting g_tup #: {0}".format(j))
        g_tup = out_q.get()
        idx = g_tup[0]
        gpsi[:,idx*chunksize:(idx+1)*chunksize] = g_tup[1]
        gBR[:,idx*chunksize:(idx+1)*chunksize] = g_tup[2]
        gBZ[:,idx*chunksize:(idx+1)*chunksize] = g_tup[3]

    for p in procs:
        p.join()

    return (gpsi,gBR,gBZ)

#def write_large_greens(R,Z,filename,rzdir=None,chunkbytes=1024**3,nprocs=1):
#    # default chunksize is 1GB per greens function
#    if rzdir is None:
#        rzdir = vstack((R,Z,ones(len(R)))).T
#    m,n = len(R),len(rzdir)
#    gridbytes = 8*m
#    print("chunkbytes: ", chunkbytes)
#    print("gridbytes: ", gridbytes)
#    n_perchunk = int(ceil(chunkbytes/float(gridbytes)))
#    print("n_perchunk: ", n_perchunk)
#    n_chunks = int(ceil(n/float(n_perchunk)))
#    print("n_chunks: ", n_chunks)
#    fh = tables.openFile(filename,mode="w")
#    filters = tables.Filters(complevel=5,complib='blosc')
#    gpsi_arr = fh.createCArray(fh.root,'gpsi',tables.Atom.from_dtype(R.dtype),
#            shape=(m,n),filters=filters)
#    gBR_arr = fh.createCArray(fh.root,'gBR',tables.Atom.from_dtype(R.dtype),
#            shape=(m,n),filters=filters)
#    gBZ_arr = fh.createCArray(fh.root,'gBZ',tables.Atom.from_dtype(R.dtype),
#            shape=(m,n),filters=filters)
#    for i in xrange(n_chunks):
#        print("processing chunk {0}".format(i))
#        gpsi_chunk,gBR_chunk,gBZ_chunk = compute_greens(R,Z,rzdir=rzdir[i*n_perchunk:(i+1)*n_perchunk],nprocs=nprocs)
#        print("chunk shapes: {0}, {1}, {2}".format(gpsi_chunk.shape,gBR_chunk.shape,gBZ_chunk.shape))
#        gpsi_arr[:,i*n_perchunk:(i+1)*n_perchunk] = gpsi_chunk
#        gBR_arr[:,i*n_perchunk:(i+1)*n_perchunk] = gBR_chunk
#        gBZ_arr[:,i*n_perchunk:(i+1)*n_perchunk] = gBZ_chunk
#
#    fh.close()




