from ControlRoom import units, classes #~/ControlRoom/lib/python2.7/site-packages/ControlRoom
import numpy as np
from matplotlib.path import Path
from scipy.io import netcdf
import os


class TRANSP:


    def __init__(self, fbm_file, cdf_file, verb=False, EcutL=0, EcutH=1e2):

        if verb:
            print('Distributions for %s' %fbm_file)

        if not os.path.isfile(fbm_file):
            if verb:
                print('NetCDF file %s not found' %fbm_file)
            return
        fv = netcdf.netcdf_file(fbm_file, 'r', mmap=False).variables

        spc_lbl = "".join(fv['SPECIES_1'].data)
        fbm  = 0.5*fv['F_%s' %spc_lbl].data
        E_eV = fv['E_%s' %spc_lbl].data
        mu   = fv['A_%s' %spc_lbl].data
        rhot_fbm = fv['X2D'].data

        cv = netcdf.netcdf_file(cdf_file, 'r', mmap=False).variables

        tfbm = fv['TIME'].data
        tim = cv['TIME3'].data
        jt = np.argmin(np.abs(tim - tfbm))

        self.Ti_cdf = cv['TI'].data[jt, :]
        self.nd_cdf = cv['ND'].data[jt, :]
        if 'NT' in cv.keys():
            self.nt_cdf = cv['NT'].data[jt, :]
        else:
            self.nt_cdf = np.zeros(len(self.nd_cdf))
        self.rhot_cdf = cv['X'].data[jt, :]

        Rfbm = fv['R2D'].data
        zfbm = fv['Z2D'].data

# Separatrix
        Rfbm_sep = fv['RSURF'].data[-1, :]
        zfbm_sep = fv['ZSURF'].data[-1, :]
        my_poly = zip(*(Rfbm_sep, zfbm_sep))
        sep_path = Path(my_poly)

# R,z grid; unit: cm; grid step: 0.5 cm
        self.Rmin = 100.
        self.Rmax = 230.
        self.zmin = -10.
        self.zmax =  30.
        self.dR = 0.5
        self.dz = 0.5
        nR = int((self.Rmax - self.Rmin)/self.dR) + 1
        nz = int((self.zmax - self.zmin)/self.dz) + 1

        self.fbmMap = {}

# Assign a FBM cell to each grid pixel
        Rgrid = np.linspace(self.Rmin, self.Rmax, nR)
        zgrid = np.linspace(self.zmin, self.zmax, nz)
        for jR, Rloc in enumerate(Rgrid):
            Rsq = (Rfbm - Rloc)**2
            self.fbmMap[jR] = {}
            for jz, zloc in enumerate(zgrid):
# Only points within separatrix
                if sep_path.contains_point((Rloc, zloc)):
                    d2 = Rsq + (zfbm-zloc)**2
                    jcell = np.argmin(d2)
                    self.fbmMap[jR][jz] = [jcell, fv['X2D'].data[jcell]]
                else:
                    self.fbmMap[jR][jz] = None
        E_MeV = 1e-6*E_eV
        EindL = (E_MeV <= EcutL)
        EindH = (E_MeV >= EcutH)
        if verb:
            print('Selecting %8.4f <= Energy/MeV <= %8.4f' %(EcutL, EcutH))

        fbm[:, :, EindL] = 0
        fbm[:, :, EindH] = 0
        self.fbmdata = {}
        n_rhot = len(rhot_fbm)
        for jmc in range(n_rhot):
            fs = fbm[jmc, :, :].ravel()
            self.fbmdata[jmc] = classes.TabulatedDistribution(E_eV, mu, fs, units.eV)


    def dist_dens(self, Rlos_cm, zlos_cm):

        jfbm_rho = self.closestVolume(Rlos_cm, zlos_cm)
        if jfbm_rho is not None:
            dist_fi = self.fbmdata[jfbm_rho[0]]
            n_fi = dist_fi.integral()/units.centimeters**3
            Ti1 = np.interp(jfbm_rho[1], self.rhot_cdf, self.Ti_cdf)
            nd1 = np.interp(jfbm_rho[1], self.rhot_cdf, self.nd_cdf)
            Ti = Ti1*units.eV
            nd = nd1/units.centimeters**3
            return dist_fi, n_fi, classes.Maxwellian(Ti), nd
        else:
            return None


    def closestVolume(self, Rlos_cm, zlos_cm):

        if (Rlos_cm < self.Rmin):
            return None
        elif (Rlos_cm > self.Rmax):
            return None
        elif (zlos_cm < self.zmin):
            return None
        elif (zlos_cm > self.zmax):
            return None
        else:
#            jR = int( (Rlos_cm - self.Rmin)/self.dR + 0.5 )
#            jz = int( (zlos_cm - self.zmin)/self.dz + 0.5 )
            jR = int( (Rlos_cm - self.Rmin)/self.dR )
            jz = int( (zlos_cm - self.zmin)/self.dz )
            return self.fbmMap[jR][jz]
