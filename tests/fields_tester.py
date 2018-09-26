from systems import TREXCoilSet
import matplotlib.pyplot as plt
import numpy as np
import grids
import sys
import time

n=300
grid = grids.RectGrid((0,1,n+1),(-1,1,2*n+1))
t = TREXCoilSet(10,10,nproc_list=[4,4])
t0 = time.time()
t.grid = grid
t1 = time.time()
print t1-t0
t.currents=10,10
plt.contourf(t.grid.R2D,t.grid.Z2D,np.sqrt(t.BZ**2+t.BR**2))
plt.contour(t.grid.R2D,t.grid.Z2D,t.psi)
plt.show()
t.currents=100,-100
plt.contourf(t.grid.R2D,t.grid.Z2D,np.sqrt(t.BZ**2+t.BR**2))
plt.contour(t.grid.R2D,t.grid.Z2D,t.psi)
plt.show()
#print "Adding grid and computing psi for n = {0}: {1} grid points and {2} processes: {3}".format(n,(n+1)*(2*n+1),m,t1-t0)
