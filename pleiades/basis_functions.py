from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
import scipy.special as spec


class spline:
    '''
    Master Spline Class
    defines basic data structures and basic functions
    '''
    def __init__(self, x_data, y_data):
        x_data = np.array(x_data,dtype=float)
        y_data = np.array(y_data,dtype=float)
        assert len(x_data) == len(y_data), \
            'Vectors are not of equal lengths'
        self.x_data = x_data
        self.y_data = y_data
        self.dimension = len(x_data)
        self.d_x = (x_data[1:] - x_data[0:-1])  # length = dimension-1
        self.d_y = (y_data[1:] - y_data[0:-1])  # length = dimension-1
        
    def __call__(self, x_value, deriv=0):
        return self._interpolate(x_value, deriv)
        
    def _interpolate(self, x_value, deriv=0):
        return
        
class CubicSpline(spline):
    '''
    Spline Class for Cubic Splines
    Includes Cubic Spline _interpolate function
    '''
    def __init__(self, x_data, y_data):
        spline.__init__(self, x_data, y_data)
        
    def _interpolate(self,x_value,deriv=0):
        '''Interpolate to get the functional value to x_value'''
        x_value=np.array(x_value,dtype=float)
        y_int=np.zeros_like(x_value)
        for i in range(self.k.size-1):
            if i==0:
                tmploc=(x_value<=self.x_data[i+1])*(x_value>=self.x_data[i])
            else:
                tmploc=(x_value<=self.x_data[i+1])*(x_value>self.x_data[i])
            xs=x_value[tmploc]
            if xs.size==0:continue
            x_l, x_u = self.x_data[i], self.x_data[i + 1]
            y_l, y_u = self.y_data[i], self.y_data[i + 1]
            k_l, k_u = self.k[i], self.k[i + 1]
            d_x = (x_u - x_l) 
            t=(xs-x_l)/(d_x)
            a=k_l*d_x-(y_u-y_l)
            b=-k_u*d_x+(y_u-y_l)
            if deriv == 1:
                y_int[tmploc]=(y_u-y_l)/d_x+(1.-2.*t)*(a*(1-t)+b*t)/d_x+\
                            t*(1-t)*(b-a)/d_x
            elif deriv == 2:
                y_int[tmploc]=2.*(b-2.*a+(a-b)*3.*t)/d_x**2
            else:
                y_int[tmploc]=(1.-t)*y_l+t*y_u+t*(1.-t)*(a*(1.-t)+b*t)
        return y_int 
        
class NaturalCubicSpline(CubicSpline):
    def __init__(self, x_data, y_data):
        CubicSpline.__init__(self, x_data, y_data)
        '''Convience Pointers'''
        dimension=self.dimension
        d_x=self.d_x
        d_y=self.d_y
        
        '''Define Matrix'''
        A=np.matrix(np.zeros((dimension,dimension)))
        for i in range(0,dimension-1):
            A[i,i]=2*(1/d_x[i]+1/d_x[i-1])
            A[i+1,i]=1/d_x[i]
            A[i,i+1]=1/d_x[i]
        A[0,0]=2/d_x[0]
        A[-1,-1]=2/d_x[-1]
        
        '''Define the b vector'''
        b=np.matrix(np.zeros((dimension))).T
        b[0]=3*d_y[0]/d_x[0]**2
        b[-1]=3*d_y[-1]/d_x[-1]**2
        for i in range(1,dimension-1):
            b[i]=3*(d_y[i]/d_x[i]**2+d_y[i-1]/d_x[i-1]**2)
            
        '''Solve for Slopes'''
        k=np.linalg.solve(A,b)
        self.k=np.array(k)
        
class ClampedCubicSpline(CubicSpline):
    def __init__(self, x_data, y_data,yp=[0,0]):
        CubicSpline.__init__(self, x_data, y_data)
        '''Data check'''
        assert len(yp)==2,'yp must be a vector of length 2'
        '''Convience Pointers'''
        dimension=self.dimension
        d_x=self.d_x
        d_y=self.d_y
        '''Define Matrix'''
        A=np.matrix(np.zeros((dimension-2,dimension-2)))
        for i in range(0,dimension-2):
            A[i,i]=2*(1./d_x[i]+1./d_x[i-1])
            try:
                A[i+1,i]=1/d_x[i]
                A[i,i+1]=1/d_x[i]
            except: pass
        '''Define the b vector'''
        b=np.matrix(np.zeros((dimension-2))).T
        for i in range(0,dimension-2):
            b[i]=3.*(d_y[i]/d_x[i]**2+d_y[i+1]/d_x[i+1]**2)
        b[0]+=-1.*yp[0]/d_x[0]
        b[-1]+=-1.*yp[-1]/d_x[-1]
        '''Solve for Slopes and add clamped slopes'''
        k=np.linalg.solve(A,b)
        ktmp=np.zeros(dimension)
        ktmp[0]=yp[0]
        ktmp[-1]=yp[-1]
        ktmp[1:-1]=k.T
        k=ktmp
        self.k=np.array(k)
        
class FirstClampedCubicSpline(CubicSpline):
    '''
    Class for doing special clamped splines for 
    BC's on Cylindrical Magnetic Fitting Problem
    '''
    def __init__(self, x_data, y_data,yp=0,m=0):
        CubicSpline.__init__(self, x_data, y_data)
        '''Convience Pointers'''
        dimension=self.dimension
        d_x=self.d_x
        d_y=self.d_y
        x_data=self.x_data
        y_data=self.y_data
        A=np.matrix(np.zeros((dimension-1,dimension-1)))
        for i in range(0,dimension-2):
            A[i,i]=2*(1/d_x[i]+1/d_x[i-1])
            try:
                A[i+1,i]=1/d_x[i]
                A[i,i+1]=1/d_x[i]
            except: pass
        A[-1,-2]=2/d_x[-1]
        A[-1,-1]=4/d_x[-1]+1/x_data[-1]
        b=np.matrix(np.zeros((dimension-1))).T
        for i in range(0,dimension-2):
            b[i]=3*(d_y[i]/d_x[i]**2+d_y[i+1]/d_x[i+1]**2)
        b[0]+=-1*yp/d_x[0]
        b[-1]+=6*d_y[-1]/d_x[-1]**2+m**2*y_data[-1]/x_data[-1]**2
        k=np.linalg.solve(A,b)
        ktmp=np.zeros(dimension)
        ktmp[0]=yp
        ktmp[1:]=k.T
        k=ktmp
        self.k=np.array(k)
        
class SecondClampedCubicSpline(CubicSpline):       
     def __init__(self, x_data, y_data,m=1):
        CubicSpline.__init__(self, x_data, y_data)
        '''Convience Pointers'''
        dimension=self.dimension
        d_x=self.d_x
        d_y=self.d_y
        x_data=self.x_data
        y_data=self.y_data
        A=np.matrix(np.zeros((dimension,dimension)))
        for i in range(0,dimension-1):
            A[i,i]=2*(1/d_x[i]+1/d_x[i-1])
            A[i+1,i]=1/d_x[i]
            A[i,i+1]=1/d_x[i]
        A[0,0]=2
        A[0,1]=1
        A[-1,-2]=2/d_x[-1]
        A[-1,-1]=4/d_x[-1]+1/x_data[-1]
        b=np.matrix(np.zeros((dimension))).T
        for i in range(1,dimension-1):
            b[i]=3*(d_y[i]/d_x[i]**2+d_y[i-1]/d_x[i-1]**2)
        b[0]+=3*y_data[1]/x_data[1]
        b[-1]+=6*d_y[-1]/d_x[-1]**2+m**2*y_data[-1]/x_data[-1]**2
        k=np.linalg.solve(A,b)
        self.k=np.array(k)

class QuarticSpline(spline):
    '''
    Spline Class for Quartic Splines
    Includes Quartic Spline _interpolate function
    '''
    def __init__(self, x_data, y_data):
        spline.__init__(self, x_data, y_data)
        assert self.d_x.std()<1e-12, \
            'x_data must be equally spaced for Quartic Splines'
        self.d_x=self.d_x.mean()
    def _interpolate(self,x_value,deriv=0):
        deriv=int(deriv)
        '''Interpolate to get the functional value to x_value'''
        x_value=np.array(x_value,dtype=float)
        y_int=np.zeros_like(x_value)
        for i in range(1,self.z.size):
            if i==1:
                tmploc=(x_value<=self.x_data[i])*(x_value>=self.x_data[i-1])
            else:
                tmploc=(x_value<=self.x_data[i])*(x_value> self.x_data[i-1])
            xs=x_value[tmploc]
            if xs.size==0:continue
            x_l, x_u = self.x_data[i-1], self.x_data[i]
            y_l, y_u = self.y_data[i-1], self.y_data[i]
            z_l, z_u = self.z[i-1], self.z[i]
            C        = self.C[i-1]
            d_x = self.d_x
            if deriv == 0:
                y_int[tmploc]=(z_u/(24*d_x)*(xs-x_l)**4
                              -z_l/(24*d_x)*(x_u-xs)**4
                              +(-z_u/24*d_x**2+y_u/d_x)*(xs-x_l)
                              +( z_l/24*d_x**2+y_l/d_x)*(x_u-xs)
                              +C*(xs-x_l)*(x_u-xs))
            elif deriv == 1:
                y_int[tmploc]=(z_u/(6*d_x)*(xs-x_l)**3
                              +z_l/(6*d_x)*(x_u-xs)**3
                              +(-z_u/24*d_x**2+y_u/d_x)
                              -( z_l/24*d_x**2+y_l/d_x)
                              +C*(x_l+x_u-2*xs))
            elif deriv == 2:
                y_int[tmploc]=(z_u/(2*d_x)*(xs-x_l)**2
                              -z_l/(2*d_x)*(x_u-xs)**2
                              -2*C)
            elif deriv == 3:
                y_int[tmploc]=(z_u/(d_x)*(xs-x_l)
                              +z_l/(d_x)*(x_u-xs))
            elif deriv == 4:
                y_int[tmploc]=(z_u/(d_x)-z_l/(d_x))
        return y_int 


basis_map = {"natural_cubic_spline":NaturalCubicSpline,
        "clamped_cubic_spline":ClampedCubicSpline,
        "clamped_cubic_spline1":FirstClampedCubicSpline,
        "clamped_cubic_spline2":SecondClampedCubicSpline,
        "quartic_spline":QuarticSpline,
        "poly":np.polynomial.polynomial.Polynomial,
        "j0":spec.j0,
        "j1":spec.j1,
        "y0":spec.y0,
        "y1":spec.y1,
        "i0":spec.i0,
        "i1":spec.i1,
        "k0":spec.k0,
        "k1":spec.k1,
        }

def build_basis(basis='poly',**kwargs):
    basis_fn = basis_map[basis.lower()]
    if basis is 'poly':
        return _poly_basis(**kwargs)
    elif 'spline' in basis:
        return _spline_basis(basis_fn,**kwargs)
    else:
        raise NotImplementedError

def _poly_basis(**kwargs):
    poly_deg = kwargs.pop("poly_deg",3)
    coeffs = kwargs.pop("coeffs",None)
    if coeffs is None:
        coeffs = np.eye(poly_deg)
    return [basis_fn(coeffs[:,i]) for i in xrange(coeffs.shape[1])]

def _spline_basis(spline_fn,**kwargs):
    n_knots = kwargs.pop("n_knots",5)
    domain = tuple(kwargs.pop("domain",(0,1)))
    xdata = kwargs.pop("xdata",None)
    ydata = kwargs.pop("ydata",None)
    if xdata is None:
        xdata = np.linspace(domain[0],domain[1],n_knots)
    if ydata is None:
        ydata = np.eye(n_knots)
    if len(xdata.squeeze().shape) == 1:
        xdata = xdata.squeeze()
        return [spline_fn(xdata,ydata[:,i],**kwargs) for i in range(ydata.shape[1])]
    else:
        return [spline_fn(xdata[:,i],ydata[:,i],**kwargs) for i in range(ydata.shape[1])]

class BasisMatrix(object):
    def __init__(self,basis,**kwargs):
        self._basis_fns = build_basis(basis=basis,**kwargs)
        self._basis_dim = len(self._basis_fns)

    @property
    def basis_fns(self):
        return self._basis_fns

    @property
    def basis_dim(self):
        return self._basis_dim

    def __call__(self,domain,**kwargs):
        arr = np.zeros((len(domain),self._basis_dim))
        for i,fn in enumerate(self._basis_fns):
            arr[:,i] = fn(domain,**kwargs)
        return arr


