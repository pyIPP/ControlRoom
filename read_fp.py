import os
from scipy.io import netcdf
import numpy as np


class READ_FP:


    def __init__(self, fp_file, spec='D', nthe_eq=201):

        print('Reading FP file %s' %fp_file)
        if not os.path.isfile(fp_file):
            print('%s not found' %fp_file)
            return

        cdf = netcdf.netcdf_file(fp_file, 'r', mmap=False)
        cv = cdf.variables

        for key in cv.keys():
            cv[key] = np.atleast_1d(cv[key].data)
#        for key, val in cdf.dimensions.items():
#            print key, val
        for key in ('Nu', 'NrdPsi', 'Nspec', 'LegMod'):
            cv[key] = cdf.dimensions[key]

# Select D or H

        if spec == 'H':
            if 1 in cv['hftyp'][:]:
                (j_spc, ) = np.where( (cv['hftyp'] == 1) & (cv['A_i'] == 1) )
            else:
                print('No H distribution function found')
                return

        if spec == 'D':
            if 2 in cv['hftyp']:
                (j_spc, ) = np.where( (cv['hftyp'] == 2) * (cv['A_i'] == 2) )
            else:
                j_spc = 0

        mass = float(cv['A_i'][j_spc])

# Mesh for pitch angle, velocity angle, rho and velocity

        n_mu = 100
        n_theta = 101
        dmu = 2./float(n_mu)
        dth = np.pi/(n_theta-1)

        self.mu_grid  = np.linspace(-1+0.5*dmu, 1-0.5*dmu, n_mu)
        self.the_grid = np.linspace(0, np.pi, n_theta)
        self.rho_grid = cv['rho']
        self.dV       = cv['dV']
        if 'nptu' in cv.keys():
            n_v = cv['nptu'][j_spc]
        else:
            n_v = cv['Nu']
        if 'Umax' in cv.keys():
            u_max = cv['Umax'][j_spc]
        elif 'umax' in cv.keys():
            u_max = cv['umax'][j_spc]
        n_rho = cv['NrdPsi']

        v_grid = np.arange(n_v)
        v_grid_sq = v_grid**2

        du  = np.squeeze(u_max/float(n_v - 1)) #velocity bin
        ni_du = np.squeeze(2*np.pi*cv['Ni'][:, j_spc])*du

# Flag: > 0 <-> ICRF on

        hf = np.sum(cv['hfpwe'][:]) + np.sum(cv['hfpwf'][:, :], axis=None)
        self.icrf_flag = (hf > 0)

# Add odd Legendre Polynomials in case of asymmetry (NBI)

        if len(cv['Addodd'][:]) > 1:
            addodd = cv['Addodd'][j_spc]
        else:
            addodd = cv['Addodd'][0]

        nodd = 2 - int(addodd)

        print('nodd = %d' %nodd)
        print('A=%d' %int(mass))

# Initialise arrays

        print n_v, n_rho
        self.uarr  = np.zeros((n_v, n_rho))
        self.Egrid = np.zeros((n_v, n_rho))
        dE = np.zeros((n_v, n_rho))
        jacob   = np.zeros((n_v, n_rho))

        f_v_mu = np.zeros((n_rho, n_v, n_mu)) #rho, v, mu
        f_v_th = np.zeros((n_rho, n_v, n_theta))
        coeffs = np.zeros((cv['LegMod'], n_v-1))

        for j_rho in range(n_rho):
            ti = cv['Ti'][j_rho, j_spc]
            ni = cv['Ni'][j_rho, j_spc]
            if u_max.ndim == 2:
                umax = u_max[j_rho]
            else:
                umax = u_max
            self.uarr[: , j_rho] = np.linspace(0, umax, n_v)
            vsq = self.uarr[: , j_rho]**2
            self.Egrid[: , j_rho] = 1e3*vsq*ti #eV
            dE[: , j_rho] = np.gradient(self.Egrid[: , j_rho])

            jacob[1:, j_rho] = 1./(1e3*self.uarr[1:, j_rho] * ti)
# FP distribution function
            for jleg in range(int(cv['LegMod']/nodd)):
                coeffs[nodd*jleg, :] = cv['Fn'][j_spc, j_rho, jleg, 1:n_v]
            f_v_mu[j_rho, 1:n_v, :] = np.polynomial.legendre.legval( \
                  self.mu_grid, coeffs )
            f_v_th[j_rho, 1:n_v, :] = np.polynomial.legendre.legval( \
                  np.cos(self.the_grid), coeffs)

        ind_v_mu = np.where(f_v_mu > 10**15)
        ind_v_th = np.where(f_v_th > 10**15)
        f_v_mu[ind_v_mu] = 0
        f_v_th[ind_v_th] = 0

# f(rho, v, theta);  Jacob(v, rho);  dE(v, rho)

        tmp_v_mu = f_v_mu.transpose(2, 1, 0) * np.outer(v_grid_sq, ni_du*du)
        self.f_v_mu = tmp_v_mu.transpose(2, 1, 0)

        tmp_E_mu  = self.f_v_mu.transpose(2, 1, 0) * jacob
        tmp_dE_mu = tmp_E_mu * dE
        self.f_mu_E = 1/mass*tmp_E_mu.transpose(2, 0, 1)
        dE_f_E_mu   = 1/mass*tmp_dE_mu.transpose(2, 1, 0)

        tmp_v_th = f_v_th.transpose(2, 1, 0) *ni_du
        self.f_v_th    = (tmp_v_th*du).transpose(2, 1, 0)
        self.f_vpa_vpe = (tmp_v_th   ).transpose(2, 1, 0)
        self.f_v_th    *= np.outer(v_grid_sq, np.sin(self.the_grid))
        self.f_vpa_vpe *= np.outer(v_grid   , np.sin(self.the_grid))

# Integral in velocity space -> n_D(rho) (normalisation check)

        self.ni  = dmu * du * np.sum(self.f_v_mu, axis=(1, 2))
        self.ni2 = dth * du * np.sum(self.f_v_th, axis=(1, 2))
        self.ni3 = dmu * np.sum(dE_f_E_mu, axis=(1, 2))

        self.ti = cv['Ti'][:, j_spc]
        self.nd = cv['Ni'][:, j_spc]
