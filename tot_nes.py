import os
crwd = os.path.dirname(os.path.realpath(__file__))


def spec_loc(A_in):

    import numpy as np
    from ControlRoom import units, classes

    nd, ti, nfast, f_cm3_eV, E_eV, mu, E_min, E_max, Ebins, samples, src = A_in
    nes = np.zeros(Ebins)
    reaction = classes.DDN3HeReaction()
    classes.Kinematics.setMode(classes.RELATIVISTIC)
    Egrid_MeV = np.linspace(E_min, E_max, Ebins)
    Egrid = Egrid_MeV*units.MeV
    nseed = 1458940
    classes.Randomizer.seed(nseed)

    if np.isnan(ti) or ti <= 0:
        return nes
    if src == 'bt':
        n1 = nd/units.centimeters**3
        f1 = classes.Maxwellian(ti*units.eV)
        f2 = classes.TabulatedDistribution(E_eV, mu, f_cm3_eV, units.eV)
        if nfast is None:
            nf_cm3 = f2.integral()
            if (nf_cm3 <= 0):
                return nes
            n2 = nf_cm3/units.centimeters**3
        else:
            n2 = nfast/units.centimeters**3
    elif src == 'bb':
        f1 = classes.TabulatedDistribution(E_eV, mu, f_cm3_eV, units.eV)
        if nfast is None:
            nf_cm3 = f1.integral()
            if nf_cm3 <= 0:
                return nes
            n1 = nf_cm3/units.centimeters**3
        else:
            n1 = nfast/units.centimeters**3
        n2 = n1
        f2 = f1
    elif src == 'th':
        n1 = nd/units.centimeters**3
        f1 = classes.Maxwellian(ti*units.eV)
        n2 = n1
        f2 = f1

    if (n1*units.centimeters**3 > 0) and (n2*units.centimeters**3 > 0):
        spec = classes.CalculateSpectrum( \
               reaction=reaction, product=classes.Neutron(), \
               f1=f1, n1=n1, \
               f2=f2, n2=n2, \
               Emin=E_min*units.MeV, Emax=E_max*units.MeV, \
               bins=Ebins, samples=samples )
        for jE, En in enumerate(Egrid):
            nes[jE] = spec[En]*units.seconds*units.centimeters**3

    return nes


def TOT_NES(inp_d, code='tr'):


    """
    The programs returns the volume integrated neutron spectrum
    from d-d -> n-He3 reactions
    separating the thermonuclear, beam-target and beam-beam contributions
    """
    import logging
    import numpy as np
    from scipy.io import netcdf
    import h5py
    from ControlRoom import units, classes
    from multiprocessing import Pool, cpu_count
    import read_fp, read_ascot

    log = logging.getLogger()

# Input

    fbm_file = inp_d['fbm']
    cdf_file = inp_d['cdf']
    asc_file = inp_d['asc']
    fp_file  = inp_d['fp']
    Ebins   = int(inp_d['n_bin'])
    E_min   = float(inp_d['Emin'])
    E_max   = float(inp_d['Emax'])
    EcutL   = float(inp_d['EcutL'])
    EcutH   = float(inp_d['EcutH'])
    reac    = inp_d['reac']
    samples = int(inp_d['n_samp'])
    code    = inp_d['code']

    print('# samples: %d' %samples)

    dE = (E_max - E_min)/float(Ebins-1)

    nes = {}

# FBM

    if code == 'tc':

        rfp = read_fp.READ_FP(fp_file)
        fp  = rfp.f_mu_E
        nes['rho'] = rfp.rho_grid
        nes['dV']  = rfp.dV
        nrho = len(rfp.rho_grid)
        nd  = rfp.ni
        mu  = rfp.mu_grid

        log.info('Calculating tot')
        pool = Pool(cpu_count())
        tot = pool.map(spec_loc, [( nd[jrho], 0.1, nd[jrho], fp[jrho, :, :].ravel(), rfp.Egrid[:, jrho], mu, E_min, E_max, Ebins, samples, 'bb') for jrho in range(nrho)] )
        pool.close()
        pool.join()
        nes['tot'] = 0.5*np.array(tot)/dE
        
        return nes

    elif code == 'tr':

        fv = netcdf.netcdf_file(fbm_file, 'r', mmap=False).variables

        spc_lbl = "".join(fv['SPECIES_1'].data)
        fbm_eV = 0.5*fv['F_%s' %spc_lbl].data  # 0.5 due to special TRANSP definition; cell, pitch, energy
        E_eV     = fv['E_%s' %spc_lbl].data
        mu       = fv['A_%s' %spc_lbl].data
        rhomc    = fv['X2D'].data
        nes['time']  = fv['TIME'].data
        nes['bmvol'] = fv['BMVOL'].data
        n_cells = len(rhomc)

# CDF

        cv = netcdf.netcdf_file(cdf_file, 'r', mmap=False).variables
        tcdf = cv['TIME3'][:]
        jt   = np.argmin(np.abs(tcdf - nes['time']))
        ti_x = cv['TI'][jt]
        nd_x = cv['ND'][jt]
        nx = len(ti_x)
        nfast = cv['BDENS2_D'][jt]
        ti_cell = []
        nd_cell = []
        for j_cell in range(n_cells):
            xrho = rhomc[j_cell]
            jrho = np.argmin(np.abs(cv['X'][jt] - xrho))
            ind  = [jrho-1, jrho, jrho+1]
            ti_cell.append(np.interp(xrho, cv['X'][jt, ind], ti_x[ind]))
            nd_cell.append(np.interp(xrho, cv['X'][jt, ind], nd_x[ind]))

        EindL = (E_eV <= 1e6*EcutL) #eV
        EindH = (E_eV >= 1e6*EcutH) #eV

        print('Selecting %8.4f <= Energy/MeV <= %8.4f' %(EcutL, EcutH))

        fbm_eV[:, :, EindL] = 0
        fbm_eV[:, :, EindH] = 0

        log.info('Calculating BT')
        print(nd_cell[0], ti_cell[0], nfast[0], fbm_eV[0, :, :].ravel(), E_eV, mu)
        pool = Pool(cpu_count())
        bt = pool.map(spec_loc, [( nd_cell[jc], ti_cell[jc], nfast[jc], fbm_eV[jc, :, :].ravel(), E_eV, mu, E_min, E_max, Ebins, samples, 'bt') for jc in range(n_cells)] )
        pool.close()
        pool.join()
        nes['bt'] = np.array(bt)/dE

        log.info('Calculating BB')
        pool = Pool(cpu_count())
        bb = pool.map(spec_loc, [( nd_cell[jc], ti_cell[jc], nfast[jc], fbm_eV[jc, :, :].ravel(), E_eV, mu, E_min, E_max, Ebins, samples, 'bb') for jc in range(n_cells)] )
        pool.close()
        pool.join()
        nes['bb'] = 0.5*np.array(bb)/dE

        log.info('Calculating TH')
        pool = Pool(cpu_count())
        th = pool.map(spec_loc, [( nd_x[jx], ti_x[jx], 0, 0, E_eV, mu, E_min, E_max, Ebins, samples, 'th') for jx in range(nx)] )
        pool.close()
        pool.join()
        nes['th'] = 0.5*np.array(th)/dE

    elif code == 'asc':

        asc = read_ascot.ASCOT(asc_file)

        fbm_tmp = asc.fbm[0, 0, :, ::-1, :, :, 0] #energy, pitch, z, R [MeV, cm**3]

        nes['rhop']  = asc.rho_p 
        nx = len(asc.rho_p)
        nD = asc.nD # cm**-3
        nE, n_p, nz, nR = fbm_tmp.shape
        nes['dvol']  = asc.dvol
        nes['bmvol'] = np.tile(asc.bmvol, nz)
        n_cells = nR*nz
        ti_cell = np.zeros(n_cells)
        nd_cell = np.zeros(n_cells)
        fbm_cell = np.zeros((nE, n_p, n_cells))

        for jR, R in enumerate(asc.grid['R']):
            for jz, z in enumerate(asc.grid['z']):
                xrho = asc.rho_pol[jR, jz]
                jrho = np.argmin(np.abs(asc.rho_p - xrho))
                ind  = [jrho-1, jrho, jrho+1]
                ti_cell[jz*nR + jR] = np.interp(xrho, asc.rho_p[ind], asc.ti[ind])
                nd_cell[jz*nR + jR] = np.interp(xrho, asc.rho_p[ind], nD[ind])
                fbm_cell[:, :, jz*nR + jR] = fbm_tmp[:, :, jz, jR]

        fbm_eV = 1.602e-19*fbm_cell.swapaxes(0, 2) # cell, pitch, energy, like TRANSP
        E_eV = asc.grid['energy']*6.242e+18
        EindL = (E_eV <= 1e6*EcutL) #eV
        EindH = (E_eV >= 1e6*EcutH) #eV

        mu    = asc.grid['pitch']

        print('Selecting %8.4f <= Energy/MeV <= %8.4f' %(EcutL, EcutH))
        fbm_eV[: , :, EindL] = 0
        fbm_eV[: , :, EindH] = 0

        log.info('Calculating BT')
        pool = Pool(cpu_count())
        bt = pool.map(spec_loc, [( nd_cell[jc], ti_cell[jc], None, fbm_eV[jc, :, :].ravel(), E_eV, mu, E_min, E_max, Ebins, samples, 'bt') for jc in range(n_cells)])
        pool.close()
        pool.join()
        nes['bt'] = np.array(bt)/dE

        log.info('Calculating BB')
        pool = Pool(cpu_count())
        bb = pool.map(spec_loc, [( nd_cell[jc], ti_cell[jc], None, fbm_eV[jc, :, :].ravel(), E_eV, mu, E_min, E_max, Ebins, samples, 'bb') for jc in range(n_cells)] )
        pool.close()
        pool.join()
        nes['bb'] = 0.5*np.array(bb)/dE

        log.info('Calculating TH')
        pool = Pool(cpu_count())
        th = pool.map(spec_loc, [( nD[jx], asc.ti[jx], 0, 0, E_eV, mu, E_min, E_max, Ebins, samples, 'th') for jx in range(nx)] )
        pool.close()
        pool.join()
        nes['th'] = 0.5*np.array(th)/dE


    log.info('Finished')

    return nes


if __name__ == '__main__':

    import os, pickle
    f_setup = '%s/nes.pkl' %crpy_dir

    f = open(f_setup, 'rb')
    setup_d = pickle.load(f)
    f.close()

    nes = TOT_NES(setup_d)
