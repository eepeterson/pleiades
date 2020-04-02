from abc import ABCMeta
import numpy as np


class Grid(metaclass=ABCMeta):
    def __init__(self):
        pass


class PointsListGrid(Grid):
    def __init__(self, rpts, zpts):
        self._R = np.asarray(rpts).squeeze()
        self._Z = np.asarray(zpts).squeeze()

    @property
    def shape(self):
        return self.R.shape

    @property
    def R(self):
        return self._R

    @property
    def Z(self):
        return self._Z

    @property
    def r(self):
        return np.sqrt(self._R**2 + self._Z**2)

    @property
    def theta(self):
        return np.cos(self._Z/self.r)


class RChord(PointsListGrid):
    def __init__(self, rpts, z0=0.):
        zpts = z0*np.ones(len(rpts))
        super().__init__(rpts, zpts)

class ZChord(PointsListGrid):
    def __init__(self, zpts, r0=0.):
        rpts = r0*np.ones(len(zpts))
        super().__init__(rpts, zpts)

class SphericalRChord(PointsListGrid):
    def __init__(self, rpts, z0=0.):
        zpts = z0*np.ones(len(rpts))
        super().__init__(rpts, zpts)

class ThetaChord(PointsListGrid):
    def __init__(self, rpts, z0=0.):
        zpts = z0*np.ones(len(rpts))
        super().__init__(rpts, zpts)


class RectGrid(Grid):
    def __init__(self, rmin=0., rmax=1., nr = 101, zmin=0., zmax=1., nz = 101):
        r, z = np.linspace(rmin, rmax, nr), np.linspace(zmin, zmax, nz)
        R, Z = np.meshgrid(r, z)
        self._R = R
        self._Z = Z

    @property
    def R(self):
        return self._R

    @property
    def Z(self):
        return self._Z

    @property
    def shape(self):
        return self.R.shape

    @property
    def r(self):
        return np.sqrt(self._R**2 + self._Z**2)

    @property
    def theta(self):
        return np.cos(self._Z/self.r)


#def get_chebnodes(nx,Lx):
#    n = np.ceil((np.sqrt(8*nx+1) - 1)/2.0)
#    if np.mod(np.ceil(n/2),2) == 0.0:
#        if np.mod(n,2) == 0.0:
#            n+=1
#        else:
#            n+=2
#    else:
#        pass
#    n = int(n)
#    zlist = []
#    for i in range(1,n+1):
#        zlist.extend([z for z in us_roots(i)[0]])
#    return Lx*np.array(sorted(zlist))
#
#
#def stretched_grid(nx,Lx,kfac):
#    x = np.zeros(nx)
#    dx = np.zeros(nx)
#    dx1 = (kfac-1)/(kfac**(nx-1)-1)*Lx
#    dx[1] = dx1
#    x[1] = dx1
#    for i in range(2,nx):
#        dx[i] = kfac*dx[i-1]
#        x[i] = x[i-1] + dx[i]
#    return np.abs(x-Lx)[::-1]
