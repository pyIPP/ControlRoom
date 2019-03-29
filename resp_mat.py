import os, datetime
import numpy as np
import rw_for
from scipy.io import netcdf


def write_hepro(rm_d, fout='ddnpar.asc', f_cdf=None):

    f = open(fout, 'w')
    f.write('  %13.7e\n' %rm_d['EKA'])
    n_En, n_spc = rm_d['spc'].shape
    for jEn in range(n_En):
        f.write('  %11.5f       %5d  %11.5f  %11.5f\n' %(rm_d['En'][jEn], n_spc, rm_d['EpB'][0], rm_d['EpB'][-1]))
        spc_str = rw_for.wr_for(rm_d['spc'][jEn, :], fmt=' %13.6e', n_lin=6)
        f.write(spc_str)
    f.close()
    print('Written %s' %fout)
        

def cdf2hepro(f_cdf):

    cv = netcdf.netcdf_file(f_cdf, 'r', mmap=False).variables
    eka = cv['E_light'][1] -  cv['E_light'][0]
    hep_d = {'EKA': cv['EKA'][:], 'spc': cv['ResponseMatrix'].data, \
             'En': cv['E_NEUT'][:], 'EpB': cv['E_light_B'][:]}
    fpath, ext = os.path.splitext(f_cdf)
    write_hepro(hep_d, fout='%s.hep' %fpath)


def rsp2cdf(fspc, f_cdf='rm.cdf', verb=False, src='nresp'):

    if src == 'nresp':
        rsp = READ_NRESP(f_spc=fspc)
        En_grid = 1.e-3*np.array(rsp.En) # keV->MeV
        En_wid  = 1.e-3*np.array(rsp.En_wid)
    else:
        rsp = HEPRO(fin=fspc)
        En_grid = rsp.dE # both in MeV
        En_wid  = rsp.en2 - rsp.en1
    nEn = len(En_grid)

    len_pos = []
    for jEn in range(nEn):
        spc = rsp.spec[jEn, :]
        if spc.any() > 0:
            (ind_pos, ) = np.where(spc > 0.)
            len_pos.append(ind_pos[-1])
        else:
            len_pos.append(0)
    jEn_max = np.argmax(len_pos)

    NMAX = min( len_pos[jEn_max] + 1, len(rsp.spec[jEn_max, :]) ) # can get down by one

    En_B = np.zeros(nEn + 1)
    En_B[1:-1] = 0.5*(En_grid[1:] + En_grid[:-1])
    En_B[0]    = 0.5*(En_grid[0]  - En_grid[1]  ) + En_grid[0]
    En_B[-1]   = 0.5*(En_grid[-1] - En_grid[-2] ) + En_grid[-1]

    Ep_B = rsp.eka*np.arange((NMAX + 1))
    Ep_grid = 0.5*(Ep_B[1:] + Ep_B[:-1])

    if verb:
        print(NMAX, Ep_B[NMAX], rsp.spec[-1, NMAX-1], rsp.spec[-1, NMAX])

# NetCDF output

    f = netcdf.netcdf_file(f_cdf, 'w', mmap=False)

    f.history = "Created " + datetime.datetime.today().strftime("%d/%m/%y")

    f.createDimension('E_NEUT' , nEn)
    f.createDimension('E_light', NMAX)
    f.createDimension('E_light_B', NMAX+1)
    f.createDimension('Eka_dim', 1)

    En = f.createVariable('E_NEUT', np.float32, ('E_NEUT', ))
    En[:] = En_grid
    En.units = 'MeV'
    En.long_name = 'Neutron energy'

    Ewid = f.createVariable('En_wid', np.float32, ('E_NEUT', ))
    Ewid[:] = En_wid
    Ewid.units = 'MeV'
    Ewid.long_name = 'NRESP-energy width for each En'

    Ep = f.createVariable('E_light', np.float32, ('E_light', ))
    Ep[:] = Ep_grid
    Ep.units = 'MeVee'
    Ep.long_name = 'Equivalent photon energy grid'

    EpB = f.createVariable('E_light_B', np.float32, ('E_light_B', ))
    EpB[:] = Ep_B
    EpB.units = 'MeVee'
    EpB.long_name = 'Equivalent photon energy bins'

    Eka = f.createVariable('EKA', np.float32, ('Eka_dim', ))
    Eka[:] = rsp.eka 
    Eka.units = 'MeVee'
    Eka.long_name = 'Step for PHS bins'

    rm = f.createVariable('ResponseMatrix', np.float32, ('E_NEUT', 'E_light'))
    rm.units = '1/(s MeVee)'
    rm.long_name = 'Response functions for several neutron energies'
    rm[:] = rsp.spec[:, :NMAX]

    f.close()
    print('Stored %s' %f_cdf)


class READ_NRESP:


    def __init__(self, f_spc='/afs/ipp/home/n/nesp/nresp/inc/SPECT_MPI.DAT'):


        self.spc_d = {}
        self.tof_d = {}

        print('Reading file %s' %f_spc)

        f = open(f_spc,'r')
        lines = f.readlines()
        f.close()

        seka = lines[0]
        self.eka = float(seka)

        self.En     = []
        self.En_wid = []
        self.spc_d  = []
        self.count  = []
        sEn_list = []

        jEn = -1
        for lin in lines[1:]:
            slin = lin.strip()
            if (slin == ''):
                continue
            sarr = slin.split()
            try:
                tmp = float(sarr[0])
                for snum in sarr:
                    self.spc_d[jEn][lbl].append(float(snum))
            except:
                lbl, sEn, sEn_wid, snr = slin.split()
                if sEn not in sEn_list:
                    jEn += 1
                    self.En.append(float(sEn))
                    self.En_wid.append(float(sEn_wid))
                    self.spc_d.append({})
                    self.count.append({})
                    sEn_list.append(sEn)
                self.spc_d[jEn][lbl] = []
                if lbl[:5] == 'PP3AS':
                    self.count[jEn][lbl] = 0
                else:
                    self.count[jEn][lbl] = int(snr)

        nEn = len(self.En)
        nEp = len(self.spc_d[nEn-1]['H(N,N)H'])
        self.spec = np.zeros((nEn, nEp))

        for jEn in range(nEn):
            for lbl, arr in self.spc_d[jEn].items():
                if lbl[:5] != 'PP3AS':
                    myarr = np.zeros(nEp)
                    nloc = len(arr)
                    myarr[:nloc] = arr[:nloc]
                    self.spec[jEn, :] += myarr


class HEPRO:


    def __init__(self, fin='/afs/ipp/home/n/nesp/tofana/responses/simresp.rsp'):

        f = open(fin, 'r')

        data = []
        for line in f.readlines():
            data += line.split()
        f.close()

        dsim = np.array(data, dtype = np.float32)
        n_dsim = len(dsim)
        print('%d' %n_dsim)
        ndim = []
        dE = []
        en1 = []
        en2 = []
        spc = []
        self.eka = dsim[0]
        j = 1
        while j < n_dsim:
            dE.append(dsim[j])
            nsize = dsim[j+1]
            ndim.append(nsize)
            en1.append(dsim[j+2])
            en2.append(dsim[j+3])
            j += 4
            spc.append(dsim[j:j+nsize])
            j += nsize
        self.dE   = np.array(dE, dtype=np.float32)
        self.en1  = np.array(en1, dtype=np.float32)
        self.en2  = np.array(en2, dtype=np.float32)
        self.ndim = np.array(ndim, dtype=np.int32)
        nEn = len(self.dE)
        nEp = np.max(self.ndim)
        self.spec = np.zeros((nEn, nEp), dtype=np.float32)
        for jEn, phs in enumerate(spc):
            nphs = len(phs)
            self.spec[jEn, :nphs] = phs[:nphs]


if __name__ == "__main__":

    import matplotlib.pylab as plt
    import fconf

    rsp = HEPRO()

    wplot = True
    if wplot:
        fig1 = plt.figure(1, figsize=(20,10))
        fig1.subplots_adjust(left=0.05,bottom=0.05, right=0.98, top=0.96, \
                            hspace=0.2, wspace=0.1)
        n_cols = 6
        n_rows = 5
        nplots = n_rows*n_cols
        for jplot in range(nplots):
            ax = fig1.add_subplot(n_rows, n_cols, jplot+1)
            ax.set_xlabel('#Channel')
            ax.plot(range(int(rsp.ndim[jplot])), rsp.spec[jplot], 'b-')
            ax.text(.5, .85, '%s [MeV]' %rsp.en2[jplot], ha='center', transform=ax.transAxes)

        fig1.canvas.mpl_connect('button_press_event', fconf.on_click)

        plt.show()
