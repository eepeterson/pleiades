import numpy as np
from fields_math import compute_greens
import itertools
from multiprocessing import Pool, sharedctypes
import time


# shape of greens matrix should be
# ngreens x len(rz_pts) break up rz_pts into chunks to send to process

r, z = np.linspace(.9, 1.1, 11), np.linspace(.9, 1.1, 11)
rzw = np.array([(ri, zi, 1.) for ri in r for zi in z])

r, z = np.linspace(0, 2, 501), np.linspace(-2, 2, 1001)
rz_pts = np.array([(ri, zi) for ri in r for zi in z])

print('Serial version running...')
t0 = time.time()
gpsi, gbr, gbz = compute_greens(rzw, rz_pts)
t = time.time() - t0
print(f'Total time spent computing and returning: {t} seconds')

print('Parallel version running...')
t1 = time.time()
gpsi, gbr, gbz = compute_greens(rzw, rz_pts)
t = time.time() - t1
print(f'Total time spent computing and returning: {t} seconds')

