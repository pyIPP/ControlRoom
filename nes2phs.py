import os
import numpy as np
import resp_mat
from scipy.io import netcdf
import datetime

lbl_d = {'bt' : 'Recoil light from beam-target neutrons, per unit time and bin-width', \
         'bb' : 'Recoil light from beam-beam neutrons, per unit time and bin-width', \
         'th' : 'Recoil light from thermonuclear neutrons, per unit time and bin-width', \
         'tot': 'Recoil light from neutrons, per unit time and bin-width'}


def NES2PHS(f_nes, f_rm='/afs/ipp/home/g/git/python/neutrons/tofana/rm_bg.cdf', w_cdf=False):

    print('Using response matrix %s' %f_rm)

    nes = netcdf.netcdf_file(f_nes, 'r', mmap=False).variables
    En_MeV = 1e-3*nes['E_NEUT'].data
    phs = {}
    reactions = []
    for key in nes.keys():
        if key in ('bt', 'bb', 'th', 'tot'):
            reactions.append(key)

    fname, ext = os.path.splitext(f_rm)
    if ext.lower() == '.cdf':
        rm  = netcdf.netcdf_file(f_rm, 'r', mmap=False).variables
        nbins = len(rm['E_light'].data)
        for react in reactions:
            phs[react] = np.zeros(nbins)
            for jEn, En in enumerate(En_MeV):
                dist = (rm['E_NEUT'].data - En)**2
                jclose = np.argmin(dist)
                phs[react] += nes[react][jEn]*rm['ResponseMatrix'][jclose, :]
    else:
        rsp = resp_mat.HEPRO(fin=f_rm)
        nbins = rsp.ndim[0]
        for react in reactions:
            phs[react] = np.zeros(nbins)
            for jEn, En in enumerate(En_MeV):
                dist = (rsp.dE - En)**2
                jclose = np.argmin(dist)
                phs[react] += nes[react][jEn]*rsp.spec[jclose]


    if w_cdf:

        f_phs = f_nes.replace('nes', 'phs')

        f = netcdf.netcdf_file(f_phs, 'w', mmap=False)

        f.history = 'Created %s\n' %datetime.datetime.today().strftime("%d/%m/%y")

        f.createDimension('Ep', nbins)
        Ep = f.createVariable('Ep', np.float32, ('Ep', ))
        Ep[:] = range(len(phs[react]))
        Ep.units = 'Ch' # keVee
        Ep.long_name = 'Detector channel' #'Equivalent photon energy'

        var = {}
        for key, val in phs.items():
            var[key] = f.createVariable(key, np.float32, ('Ep',))
            var[key].units = '1/(s)'  #'1/(s keVee)'
            var[key][:] = val
            var[key].long_name = lbl_d[key]

        f.close()
        print('Stored %s' %f_phs)

    print('Finished nes2phs', f_rm)
    return phs


if __name__ == '__main__':

    import matplotlib.pylab as plt

    fnes = '/afs/ipp/home/g/git/python/ControlRoom/output/tr/29783A01_fi_1_nes_tr.cdf'

    rm1 = '/afs/ipp/home/n/nesp/tofana/responses/simresp_aug.rsp_broad'
    rm2 = '/afs/ipp/home/g/git/python/neutrons/tofana/sim_aug_gb.cdf'
    resp_mat.rsp2cdf(rm1, f_cdf=rm2, src='hep')

    phs1 = NES2PHS(fnes, f_rm=rm1)
    phs2 = NES2PHS(fnes, f_rm=rm2)
    
    plt.plot(phs1['bt'], 'g-')
    plt.plot(phs2['bt'], 'r-')
    plt.show()
