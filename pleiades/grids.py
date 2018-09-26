from numpy import array,zeros,sqrt,linspace,logspace,pi,meshgrid,sqrt

class Grid(object):
    def __init__(self):
        pass

class StructuredGrid(Grid):
    def __init__(self):
        pass

class UnstructuredGrid(Grid):
    def __init__(self):
        pass

class PointsListGrid(UnstructuredGrid):
    def __init__(self,rpts,zpts):
        R,Z = meshgrid(rpts,zpts)
        self._shape = R.shape
        self._size = R.size
        self._R = R.flatten()
        self._Z = Z.flatten()

    @property
    def shape(self):
        return self._shape
    
    @property
    def size(self):
        return self._size

    @property
    def R(self):
        return self._R.reshape(self._shape)

    @property
    def R1D(self):
        return self._R

    @property
    def Z(self):
        return self._Z.reshape(self._shape)

    @property
    def Z1D(self):
        return self._Z

    @property
    def r(self):
        return sqrt(self._R**2+self._Z**2).reshape(self._shape)

    @property
    def theta(self):
        return cos(self._Z/sqrt(self._R**2+self._Z**2)).reshape(self._shape)


class RectGrid(StructuredGrid):
    def __init__(self,(Rmin,Rmax,nR),(Zmin,Zmax,nZ)):
        R,Z = meshgrid(linspace(Rmin,Rmax,nR,dtype="float32"),linspace(Zmin,Zmax,nZ,dtype="float32"))
        self._shape = R.shape
        self._size = R.size
        self._R = R.flatten()
        self._Z = Z.flatten()

    @property
    def shape(self):
        return self._shape
    
    @property
    def size(self):
        return self._size

    @property
    def R(self):
        return self._R.reshape(self._shape)

    @property
    def R1D(self):
        return self._R

    @property
    def Z(self):
        return self._Z.reshape(self._shape)

    @property
    def Z1D(self):
        return self._Z

    @property
    def r(self):
        return sqrt(self._R**2+self._Z**2).reshape(self._shape)

    @property
    def theta(self):
        return cos(self._Z/sqrt(self._R**2+self._Z**2)).reshape(self._shape)

# things to add
# circ grid
# rectquad
# stitching
# stretched grids
#
# unstructred grids
# arb. boundary with triangles or rectquads
# higher res in certain areas

def get_chebnodes(nx,Lx):
    n = np.ceil((np.sqrt(8*nx+1) - 1)/2.0)
    if np.mod(np.ceil(n/2),2) == 0.0:
        if np.mod(n,2) == 0.0:
            n+=1
        else:
            n+=2
    else:
        pass
    n = int(n)
    zlist = []
    for i in range(1,n+1):
        zlist.extend([z for z in us_roots(i)[0]])
    return Lx*np.array(sorted(zlist))

def stretched_grid(nx,Lx,kfac):
    x = np.zeros(nx)
    dx = np.zeros(nx)
    dx1 = (kfac-1)/(kfac**(nx-1)-1)*Lx
    dx[1] = dx1
    x[1] = dx1
    for i in range(2,nx):
        dx[i] = kfac*dx[i-1]
        x[i] = x[i-1] + dx[i]
    return np.abs(x-Lx)[::-1]
