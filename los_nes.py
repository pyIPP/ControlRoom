from ControlRoom import units, classes #~/ControlRoom/lib/python2.7/site-packages/ControlRoom
import logging, pickle, os, datetime
import numpy as np
import transp, toric, ascot
from scipy.io import netcdf

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    n_threads = comm.size
    jrank = comm.rank
    mpi_flag = True
except:
    n_threads = 1
    jrank = 0
    mpi_flag = False

#name = MPI.Get_processor_name()

"""
The programs returns and plots the LOS integrated neutron spectrum
from d-d -> n-He3 reactions.
Usage:
module load impi
module load python27/mpi4py
mpirun -np 10 python los_nes.py
"""

__author__ = 'Giovanni Tardini (Tel. 1898)'
__version__ = '0.1'
__date__ = '19.02.2013'

fmt = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s')
hnd = logging.StreamHandler()
hnd.setFormatter(fmt)
logging.root.addHandler(hnd)
logging.root.setLevel(logging.INFO)
log = logging.getLogger()

lbl_d = {'bt' : 'Beam-target neutrons per unit time and energy', \
         'bb' : 'Beam-beam neutrons per unit time and energy', \
         'th' : 'Thermonuclear neutrons per unit time and energy', \
         'tot': 'Neutrons per unit time and energy'}

crpy_dir = os.path.dirname(os.path.realpath(__file__))


def los_nes():

    f_dic = '%s/nes.pkl' %crpy_dir
    f = open(f_dic, 'rb')
    setup_d = pickle.load(f)
    f.close()

    code = setup_d['code']

    verb = False
    if jrank == 0:
        verb = True

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

    if verb:
        log.info('\nStarting los_nes')
        print('# threads: %d' %n_threads)
        print('# samples per thread: %d' %samples)
        log.info('Setting profiles')
    if code == 'tr':
        f_dist = setup_d['fbm']
        profile = transp.TRANSP(f_dist, cdf_file, EcutL=EcutL, EcutH=EcutH, verb=verb)
    elif code == 'asc':
        f_dist = setup_d['asc']
        profile = ascot.ASCOT(f_dist, EcutL=EcutL, EcutH=EcutH, verb=verb)
    elif code == 'tc':
        f_dist = setup_d['fp']
        profile = toric.TORIC(f_dist, verb=verb)

    if verb:
        log.info('Setting cells')
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

    if verb:
        print('Reaction %s' %reac)
    reaction = classes.DDN3HeReaction()
    whichProduct = 1 # 1 <-> Neutron
    classes.Kinematics.setMode(classes.RELATIVISTIC)

# Make the randomisation reproducible, but different for each thread
    rdm = np.random.RandomState(seed=156677)
    seed_arr = rdm.random_integers(0, 10000, size=n_threads)
    nseed = int(seed_arr[jrank])
    print('Rank %d, random seed %d' %(jrank, nseed))
    classes.Randomizer.seed(nseed)

    Egrid = Egrid_MeV*units.MeV
    nes  = {}
    rate = {}
    tmp = np.zeros(Ebins)

    n_samp = samples
    for react in reactions:
        if verb:
            log.info('Calculating %s' %react)
            print('# of valid %s-cells: %d' %(react, len(cells[react]) ))
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

        if mpi_flag:
            comm.Reduce(spec, tmp, op=MPI.SUM)
            nes[react] = tmp/n_threads
        else:
            nes[react] = spec

    if code == 'tc':
        nes['tot'] *= 0.5
    else:
        nes['bb']  *= 0.5
        nes['th']  *= 0.5
    for react in reactions:
        rate[react] = dE*np.sum(nes[react])

    log.info('Finished los_nes calculation %d' %jrank)

    if jrank == 0:

        rate_tot = 0
        for react in reactions:
            print('Rate for %s: %12.4e Hz' %(react, rate[react]) )
            rate_tot += rate[react]
        print('Total rate in LoS: %12.4e Hz' %rate_tot ) 

        Egrid_keV = np.array(Egrid/units.keV, dtype=np.float32)
        nE = len(Egrid_keV)

# NetCDF file output

        dir_out = ('%s/output/%s' %(crpy_dir, code))
        os.system('mkdir -p %s' %dir_out)
        ftmp  = os.path.basename(f_dist)
        fname, ext = os.path.splitext(ftmp)
        fcdf  = ftmp.replace(ext, '_nes_%s.cdf' %code)
        cdf_out = ('%s/%s' %(dir_out, fcdf))
        print(ext, code)
        print(fcdf)
        f = netcdf.netcdf_file(cdf_out, 'w', mmap=False)

        f.history = 'Created %s\n' %datetime.datetime.today().strftime("%d/%m/%y")
        f.history += 'Fast ion distribution function from file %s\n' %f_dist
        f.history += 'Cone-of-sight geometry from file %s' %setup_d['los']

        f.createDimension('E_NEUT', nE)
        En = f.createVariable('E_NEUT', np.float32, ('E_NEUT', ))
        En[:] = Egrid_keV
        En.units = 'keV'
        En.long_name = 'Neutron energy'

        var = {}
        for key, val in nes.items():
            var[key] = f.createVariable(key, np.float32, ('E_NEUT',))
            var[key].units = '1/(s keV)'
            var[key][:] = 1e-3*val
            var[key].long_name = lbl_d[key]

        f.close()
        print('Stored %s' %cdf_out)


if __name__ == '__main__':

    los_nes()
