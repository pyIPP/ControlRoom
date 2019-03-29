import sys
sys.path.append('/afs/ipp/home/g/git/python/repository')
import numpy as np
import matplotlib.pylab as plt
import fconf, rw_for
from scipy.io import netcdf

def write_hepro(rm_d, fout='ddnpar.asc', f_cdf=None):

    f = open(fout, 'w')
    f.write('  %13.7e\n' %rm_d['EKA'])
    n_En, n_spc = rm_d['spc'].shape
    for jEn in range(n_En):
        f.write('  %11.5f       %5d  %11.5f  %11.5f\n' %(rm_d['En'][jEn], n_spc, rm_d['EpB'][0], rm_d['EpB'][-1]))
        spc_str = rw_for.wr_for(rm_d['spc'][jEn, :], fmt='%13.6e', n_lin=6)
        f.write(spc_str)
    f.close()
    print('Written %s' %fout)
        

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
        self.spec = []
        self.ec = dsim[0]
        j = 1
        while j < n_dsim:
            dE.append(dsim[j])
            nsize = dsim[j+1]
            ndim.append(nsize)
            en1.append(dsim[j+2])
            en2.append(dsim[j+3])
            j += 4
            self.spec.append(dsim[j:j+nsize])
            j += nsize
        self.dE   = np.array(dE, dtype=np.float32)
        self.en1  = np.array(en1, dtype=np.float32)
        self.en2  = np.array(en2, dtype=np.float32)
        self.ndim = np.array(ndim, dtype=np.int32)




if __name__ == "__main__":


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
