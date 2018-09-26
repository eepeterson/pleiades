from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from eqmath import new_greens_test,short_greens_test


nlist = np.array([5,10,20,30,40,50])
ngrid = nlist**2
nunique = np.zeros_like(ngrid)
ngpsi = np.zeros_like(ngrid)

for i,n in enumerate(nlist):
    r,z = np.linspace(0.1,1,n),np.linspace(-.5,.5,n)
    R,Z = np.meshgrid(r,z)
#    gpsi = new_greens_test(R.flatten(),Z.flatten())
    gpsi = short_greens_test(R,Z)
    gpsi[~np.isfinite(gpsi)] = np.nan
    print(gpsi.shape)
    ngpsi[i] = len(gpsi.flatten())
    nunique[i] = len(np.unique(gpsi))
#    print(ngpsi[i])
#    print(nunique[i])
#    eps = np.finfo(float).eps
#    print(np.all(np.isclose(gpsi-gpsi.T,0,atol=10*eps)))
#    #print(np.all(gpsi[5:10,0:5]==gpsi[0:5,5:10]))
#    print(np.max(np.abs(gpsi-gpsi.T)))
#    print(np.sum(np.isclose(gpsi,0,atol=1E-12)))
    plt.matshow(gpsi)
    plt.colorbar()
    plt.show()

fig,ax = plt.subplots()
ax.plot(ngrid,nunique,"o")
ax.set_title("ngrid vs nunique")

fig1,ax1 = plt.subplots()
ax1.plot(ngpsi,nunique,"o")
ax1.set_title("ngpsi vs nunique")

fig2,ax2 = plt.subplots()
ax2.plot(ngpsi,nunique/ngpsi,"o")
ax2.set_title("ngpsi vs nunique/ngpsi")

plt.show()
