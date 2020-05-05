import os, datetime
import numpy as np
from scipy.io import netcdf


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

