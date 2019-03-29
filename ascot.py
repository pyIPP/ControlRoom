import os
from ControlRoom import units, classes #~/ControlRoom/lib/python2.7/site-packages/ControlRoom
import numpy as np
from matplotlib.path import Path
from scipy.interpolate import RectBivariateSpline
import read_ascot


class ASCOT:


    def __init__(self, asc_file, verb=False, EcutL=0, EcutH=1e2):

        if verb:
            print('Distributions for %s' %asc_file)

        if not os.path.isfile(asc_file):
            if verb:
                print('HDF5 file %s not found' %asc_file)
            return

        asc = read_ascot.ASCOT(asc_file)

        self.Rgrid_b = asc.grid['R_b']
        self.zgrid_b = asc.grid['z_b']

        E_J   = asc.grid['energy'] # J
        mu    = asc.grid['pitch']
        self.rhop_map = asc.rho_pol
        nR, nz = self.rhop_map.shape

# Distribution

        fbm_MeV = asc.fbm[0, 0, :, ::-1, :, :, 0]*1.602e-13 #energy, pitch, z, R [MeV] | cm**-3 s**-1

        E_MeV = E_J*6.242e+12
        EindL = np.where(E_MeV <= EcutL)[0]
        EindH = np.where(E_MeV >= EcutH)[0]
        if verb:
            print('Selecting %8.4f <= Energy/MeV <= %8.4f' %(EcutL, EcutH))

        fbm_MeV[EindL, :, :, :, ] = 0
        fbm_MeV[EindH, :, :, :, ] = 0
        self.fbmdata = {}
        for jR in range(nR):
            self.fbmdata[jR] = {}
            for jz in range(nz):
                fs = fbm_MeV[:, :, jz, jR].T.ravel()
                self.fbmdata[jR][jz] = classes.TabulatedDistribution(E_MeV, mu, fs, units.MeV)

# Background plasma

        self.nd_1d   = asc.nD
        self.ti_1d   = asc.ti
        self.rhop_1d = asc.rho_p


    def dist_dens(self, Rlos_cm, zlos_cm):

        jr_jz_rhop = self.closestVolume(Rlos_cm, zlos_cm)
        if jr_jz_rhop is not None:
            jR, jz = jr_jz_rhop
            if np.isnan(self.rhop_map[jR, jz]) or (self.rhop_map[jR, jz] > 1):
                return None
            dist = self.fbmdata[jR][jz]
            n_fi = dist.integral()/units.centimeters**3
            Ti1 = np.interp(self.rhop_map[jR, jz], self.rhop_1d, self.ti_1d)
            Ti = Ti1*units.eV
            nd = np.interp(self.rhop_map[jR, jz], self.rhop_1d, self.nd_1d)/units.centimeters**3
            return dist, n_fi, classes.Maxwellian(Ti), nd
        else:
            return None


    def closestVolume(self, Rlos_cm, zlos_cm):

        Rmin = self.Rgrid_b[0]
        zmin = self.zgrid_b[0]
        Rlos_m = 1e-2*Rlos_cm
        zlos_m = 1e-2*zlos_cm
        dR = self.Rgrid_b[1] - self.Rgrid_b[0] # if equidistant!
        dz = self.zgrid_b[1] - self.zgrid_b[0] # if equidistant!
# Assign a FBM cell to each point in LOS cone, via R,z grid_b
        if (Rlos_m < Rmin) or (Rlos_m > self.Rgrid_b[-1]) or \
           (zlos_m < zmin) or (zlos_m > self.zgrid_b[-1]):
            return None
        else:
            jR = int( (Rlos_m - Rmin)/dR )
            jz = int( (zlos_m - zmin)/dz )
            return jR, jz

