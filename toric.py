from ControlRoom import units, classes #~/ControlRoom/lib/python2.7/site-packages/ControlRoom
import numpy as np
import read_tc_eq, read_fp

_torus = classes.Torus(R=1.65*units.meters, k=1.6)



class TORIC:


    def __init__(self, fp_file, verb=False):


        nthe_eq = 41
        self.rfp = read_fp.READ_FP(fp_file)
        equ_file = '%seq.cdf' %fp_file[:-6]
        Rgl, zgl = read_tc_eq.read_tc_eq(equ_file, self.rfp.rho_grid, nthe_eq=nthe_eq)

        sep105 = []
        rmc   = []
        themc = []
        rhomc = []

        nrho, nthe_eq = Rgl.shape

        for jrho in range(nrho):
            for jthe in range(nthe_eq):
                R = Rgl[jrho, jthe] * units.meters
                Z = zgl[jrho, jthe] * units.meters
                rdim, _, theta = _torus.toToroidal(R, 0.0*units.meters, Z)
                r_m = float(rdim/units.meters)
                rmc.append(r_m)
                themc.append(theta)
                rhomc.append(self.rfp.rho_grid[jrho])
                if (jrho == nrho - 1):
                    sep105.append([theta, r_m])

        sep105.sort()
        the_sep, r_sep = zip(*sep105)
        the_sep = (-1.01*np.pi,) + the_sep + (1.01*np.pi,)
        r_sep   = (r_sep[0],)    + r_sep   + (r_sep[-1],)
        rMax105 = max(r_sep)
        self.bdy105 = classes.LinearInterpolation(*[the_sep, r_sep])

        rmc   = np.array(rmc)
        themc = np.array(themc)

        nr   = 201
        nthe = 721
        self.fbmMap = {}
        self.dthe = 2*np.pi/(nthe - 1.)
        self.dr = rMax105/(nr - 1.)

        r_loc   = np.linspace(0, rMax105, nr)
        the_loc = np.linspace(-np.pi, np.pi, nthe)

        for jr, rloc in enumerate(r_loc):
            self.fbmMap[jr] = [None]*nthe
            mysq = rloc**2 + rmc**2
            tmp = 2*rloc*rmc
            for jthe, theloc in enumerate(the_loc):
                if rloc < self.bdy105(theloc):
                    d2 = mysq - tmp*np.cos(themc - theloc)
                    jmin = np.argmin(d2)
                    self.fbmMap[jr][jthe] = rhomc[jmin]

        fbm = self.rfp.f_mu_E
        E   = self.rfp.Egrid # eV
        mu  = self.rfp.mu_grid
        rho = self.rfp.rho_grid

        if verb:
            print('fbm density = %12.4e' %np.max(self.rfp.ni))
        self.fbmdata = {}

        for jrho, rholoc in enumerate(rho):
            fs = fbm[jrho, :, :].ravel()
            self.fbmdata[rholoc] = classes.TabulatedDistribution(E[:, jrho], mu, fs, units.eV)


    def closestVolume(self, r_unit, the):

        r = float(r_unit/units.meters)
        if r < self.bdy105(the):
            jr   = int(r/self.dr + 0.5)
            jthe = int((the + np.pi)/self.dthe + 0.5)
            return self.fbmMap[jr][jthe]
        else:
            return None


    def dist_dens(self, x, y, z):

        r, phi, theta = _torus.toToroidal(x, y, z)
        rho = self.closestVolume(r, theta)
        if rho is not None:
            dist = self.fbmdata[rho]
#            nd = self.fbmdata[rho].integral()/units.centimeters**3
            nd = np.interp(rho, self.rfp.rho_grid, self.rfp.ni)/units.centimeters**3
            return dist, nd
        else:
            return None
