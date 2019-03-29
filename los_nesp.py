from ControlRoom import units, classes #~/ControlRoom/lib/python2.7/site-packages/ControlRoom
import logging, pickle, os
import numpy as np
import transp, toric, ascot
from scipy.io import netcdf


"""
The programs returns and plots the LOS integrated neutron spectrum
from d-d -> n-He3 reactions.
Usage:
module load anaconda
"""

__author__ = 'Giovanni Tardini (Tel. 1898)'
__version__ = '0.1'
__date__ = '19.02.2019'


def los_nes(setup_d):

    code = setup_d['code']

    cdf_file = setup_d['cdf']
    los_file = setup_d['los']
    EcutL    = float(setup_d['EcutL'])
    EcutH    = float(setup_d['EcutH'])
    reac = setup_d['reac']
    code = setup_d['code']

    Emin  = float(setup_d['Emin'])
    Emax  = float(setup_d['Emax'])
    Ebins = int(setup_d['n_bin'])
    samples = int(setup_d['n_samp'])/n_threads

    Egrid_MeV = np.linspace(Emin, Emax, Ebins)
    dE = (Egrid_MeV[-1] - Egrid_MeV[0])/float(Ebins-1)

    if code == 'tr':
        f_dist = setup_d['fbm']
        profile = transp.TRANSP(f_dist, cdf_file, EcutL=EcutL, EcutH=EcutH, verb=False)
    elif code == 'asc':
        f_dist = setup_d['asc']
        profile = ascot.ASCOT(f_dist, EcutL=EcutL, EcutH=EcutH, verb=False)
    elif code == 'tc':
        f_dist = setup_d['fp']
        profile = toric.TORIC(f_dist, verb=False)

    try:
        x_m, y_m, z_m, C, V_m3, u1, v1, w1 = np.loadtxt(los_file, comments='/', unpack=True, dtype=float)
    except IOError:
        raise

    u = [float(x) for x in u1]
    v = [float(x) for x in v1]
    w = [float(x) for x in w1]
    omega = 4*np.pi*C/V_m3*units.steradians
    x, y, z = x_m*units.meters, y_m*units.meters, z_m*units.meters
    V = V_m3*units.meters**3
    R_cm = 100.*np.hypot(x_m, y_m)
    z_cm = 100.*z_m
#        v_fluid = classes.Vector(2e5, 0, 0)*units.meter/second
    v_fluid = classes.Vector(0, 0, 0)*units.meters/units.second

    cells = {}
    if code == 'tc':
        reactions = ('tot', )
    else:
        reactions = ('bt', 'th', 'bb')
    for react in reactions:
        cells[react] = []

    for i in range(len(x_m)):

        posx = classes.Vector.Cartesian(x[i], y[i], z[i])
        posv = -classes.Vector.Cartesian(u[i], v[i], w[i]).versor

        if code == 'tc':

            dist_dens = profile.dist_dens(x[i], y[i], z[i])
            if dist_dens is not None:
                f1, n1 = dist_dens
                cells['tot'].append( \
                    classes.Cell(f1=f1, f2=f1, n1=n1, n2=n1, \
                    position=posx, volume=V[i], \
                    solidAngle=omega[i], lineOfSight=posv, weight=1.0) )

        else:

            dist_dens = profile.dist_dens(R_cm[i], z_cm[i])

            if dist_dens is not None:
                f_fi, n_fi, f_mxw, nd = dist_dens
                cells['bt'].append( \
                    classes.Cell(f1=f_fi, f2=f_mxw, n1=n_fi, n2=nd, \
                    position=posx, volume=V[i], \
                    solidAngle=omega[i], lineOfSight=posv, weight=1.0) )

                cells['th'].append( \
                    classes.Cell(f1=f_mxw, f2=f_mxw, n1=nd, n2=nd, \
                    position=posx, volume=V[i], \
                    solidAngle=omega[i], lineOfSight=posv, weight=1.0) )

                cells['bb'].append( \
                    classes.Cell(f1=f_fi, f2=f_fi, n1=n_fi, n2=n_fi, \
                    position=posx, volume=V[i], \
                    solidAngle=omega[i], lineOfSight=posv, weight=1.0) )

    reaction = classes.DDN3HeReaction()
    whichProduct = 1 # 1 <-> Neutron
    classes.Kinematics.setMode(classes.RELATIVISTIC)

# Make the randomisation reproducible, but different for each thread
    rdm = np.random.RandomState(seed=156677)
    seed_arr = rdm.random_integers(0, 10000, size=jrank+1)
    nseed = seed_arr[-1]
    print('Rank %i, random seed %i' %(jrank, nseed))
    classes.Randomizer.seed(int(nseed))

    Egrid = Egrid_MeV*units.MeV
    nes  = {}

    tmp = np.zeros(Ebins)

    n_samp = samples
    for react in reactions:

# Reducing accuracy for beam-beam, thermonuclear
        if react in ('bb', 'th'):
            n_samp = samples/5
        spc = classes.CalculateVolumeSpectrum( \
              reaction, whichProduct, cells[react], \
              Emin*units.MeV, Emax*units.MeV, Ebins, n_samp, \
              E1range = (0.0*units.MeV, 1.0*units.MeV), \
              E2range = (0.0*units.MeV, 1.0*units.MeV), \
              Bdirection = classes.Clockwise, \
              vCollective = v_fluid)

        spec = np.zeros(Ebins)
        for jE, En in enumerate(Egrid):
            spec[jE] = spc[En]*units.seconds
        spec /= dE

        nes[react] = spec

    if code == 'tc':
        nes['tot'] *= 0.5
    else:
        nes['bb']  *= 0.5
        nes['th']  *= 0.5

