from __future__ import print_function, division
import numpy as np
import matplotlib.pyplot as plt
from eqmath import new_greens_test,short_greens_test,get_gpsi


nlist = np.array([257])

for i,n in enumerate(nlist):
    r,z = np.linspace(0.1,1,n),np.linspace(-.5,.5,n)
    R,Z = np.meshgrid(r,z)
    gpsi = get_gpsi(R,Z)
#    gpsi2 = new_greens_test(R.flatten(),Z.flatten())
#    print(np.all(np.isclose(gpsi-gpsi2,0,atol=1E-16)))
#    fig,ax = plt.subplots()
#    cf = ax.matshow(gpsi-gpsi2)
#    plt.colorbar(cf)
#    fig1,ax1 = plt.subplots()
#    cf1 = ax1.matshow(gpsi2)
#    plt.colorbar(cf1)
#
#    plt.show()

