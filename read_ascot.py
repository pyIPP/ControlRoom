import os
import h5py
import numpy as np
from scipy.interpolate import RectBivariateSpline


class ASCOT:


    def __init__(self, f_in):


        f = h5py.File(f_in, 'r')

        print('\nSpecies\n')
        spec = f['/species/testParticle']
        self.mass   = spec['anum'][:]
        self.charge = spec['charge'][:]
        
        print('\nDistributions\n')

        print('\nPlasma\n')
        pl1d = f['/plasma/1d']

        dist_rho = f['/distributions/rhoDist']
        rhop_b = dist_rho['abscissae']['dim1'][:]
        self.dvol  = dist_rho['shellVolume'][:]
        self.rho_p = 0.5*(rhop_b[1:] + rhop_b[:-1])
        self.ti = np.interp(self.rho_p, pl1d['rho'][:], pl1d['ti'][:])  # eV
        self.nD = 1e-6*np.interp(self.rho_p, pl1d['rho'][:], pl1d['ni'][:, 0]) # cm**-3

        print('Vol', np.sum(self.dvol)) # m**3
        dist_grids = f['/distributions/rzPitchEdist/abscissae']

        self.grid = {}
        unit_d = {}
        for key, var in dist_grids.items():
            darr = var[:]
            lbl = '%s' %(var.attrs['name'])
            unit = '%s' %(var.attrs['unit'])
            unit_d[lbl] = unit
            dvar        = np.diff(darr)
            self.grid[lbl+'_b'] = darr
            self.grid[lbl]      = darr[:-1] + 0.5*dvar
        dR = self.grid['R_b'][1] - self.grid['R_b'][0]
        dz = self.grid['z_b'][1] - self.grid['z_b'][0]
        self.bmvol = 2.*np.pi*self.grid['R']*dR*dz
        self.fbm = 1e-6*f['/distributions/rzPitchEdist/ordinate'][:] # m**-3->cm**-3
        fbm_descr = f['/distributions/rzPitchEdist/ordinates']

        print('\nBfields\n')
 
        bf = f['bfield']

        (self.Raxis, ) = bf['raxis']
        (self.zaxis, ) = bf['zaxis']
        self.R_psi = bf['r'][:]
        self.z_psi = bf['z'][:]

        self.psi = f['bfield/2d/psi'][:]
        
        f.close()
        
# Reduce psi to FBM grid
        
        nR = len(self.grid['R'])
        nz = len(self.grid['z'])

# Bilinear interpolation

        self.psi_sep = 0
        self.psi_axis = RectBivariateSpline(self.z_psi, self.R_psi, self.psi)(self.Raxis, self.zaxis)

        psi_red = RectBivariateSpline(self.z_psi, self.R_psi, self.psi)(self.grid['R'], self.grid['z'], grid=True)

        psi_norm = (psi_red - self.psi_axis)/(self.psi_sep - self.psi_axis)
        self.rho_pol = np.sqrt(psi_norm)

#        jzmax, jRmax = np.unravel_index(psi.argmax(), psi.shape)
#        print('Rmax_psi=%8.4f, zmax_psi=%8.4f' %(R_psi[jRmax], z_psi[jzmax]))


if __name__ == '__main__':
        
    import matplotlib.pylab as plt
        
    f_in = '%s/ascot/fbm/29795_3.0s_ascot.h5' %os.getenv('HOME')
    asc = ASCOT(f_in)

    plt.figure(1, (10,12))
    rp, zp = np.meshgrid(asc.R_psi, asc.z_psi)
    levels = np.linspace(0, 0.3, 31)
    plt.contourf(rp, zp, asc.psi, levels=levels)
    plt.ylim((-1.5,1))
    plt.axis('equal')
    plt.show()
        
