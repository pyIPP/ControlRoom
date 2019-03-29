import os
import numpy as np
import mom2rz
from scipy.io import netcdf

def read_tc_eq(equ_file, rho_grid, nthe_eq=201):

    if not os.path.exists(equ_file):
        print('Equ file %s not found' %equ_file)
        return

    cv = netcdf.netcdf_file(equ_file, 'r', mmap=False).variables
    Rsurf, zsurf = mom2rz.mom2rz(cv['rmc2d'][:].T, cv['rms2d'][:].T, \
        cv['zmc2d'][:].T, cv['zms2d'][:].T, nthe=nthe_eq)

# Poloidal section, FBM contours

    Rgl = []
    zgl = []
    for j_rho, rho in enumerate(rho_grid):
        x_dist = (cv['rhopol'][:] - rho)**2
        irho = np.argmin(x_dist)
        Rgl.append(Rsurf[irho, :])
        zgl.append(zsurf[irho, :])

    return np.array(Rgl), np.array(zgl)
