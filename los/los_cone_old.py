import sys
sys.path.append('/afs/ipp/home/g/git/python/repository')
sys.path.append('/afs/ipp/aug/ads-diags/common/python/lib/')
import numpy as np
import Tkinter as tk
import los

geo_aug = {'disk_thick':0.008, 'cell_radius':0.004, \
           'cone_aper':0.42, 'det_radius':0.0254, 'tilt':0, \
           'tan_radius':0.2, 'y_det':-13.63, 'z_det':0.1, \
           'Rmaj':1.65, 'r_chamb':0.6}
tok_lbl = 'aug'


def plot_los_cone(flos='aug.los'):


    import matplotlib.pylab as plt

    print('Reading %s' %flos)

    try:
        x_m, y_m, z_m, C, V_m3, u1, v1, w1 = np.loadtxt(flos, skiprows=21, unpack=True)
    except IOError:
        raise
    R_m = np.hypot(x_m, y_m)
    print R_m

    f = plt.figure(1,figsize=(13, 5.5))
    f.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.98)

    ax1 = plt.subplot(1, 2, 1, aspect='equal')
    ax1.set_xlim([0.5, 3.5])
    ax1.set_ylim([-1.5, 1.5])
    ax1.plot(R_m, z_m, 'ro') 

    ax2 = plt.subplot(1, 2, 2, aspect='equal')
    ax2.set_xlim([-3, 3])

# Plot AUG wall
    try:
        import map_equ_20161123
        gc_r, gc_z = map_equ_20161123.get_gc()
        for key in gc_r.iterkeys():
            ax1.plot(gc_r[key], gc_z[key], 'b-')
    except:
        print('No coordinates of wall structures available for poloidal section drawing')

    try:
        import plot_aug
        dic = plot_aug.STRUCT().tor_old
        for key in dic.iterkeys():
            ax2.plot(dic[key].x, dic[key].y, 'b-')
    except:
        print('No coordinates of wall structures available for toroidal section drawing')

    ax2.plot(x_m, y_m, 'ro')
    ax2.set_xlabel('x [m]', labelpad=2)
    ax2.set_ylabel('y [m]', labelpad=-14)
    ax2.tick_params(which='major', length=4, width=0.5)

    plt.show()


class DET_LOS:


    def __init__(self):


        myframe = tk.Tk(className=' Detector line of sight')

        geo_init = geo_aug

        menuframe = tk.Frame(myframe)
        toolframe = tk.Frame(myframe)
        entframe  = tk.Frame(myframe)
        toolframe.grid(row=0, sticky=tk.W+tk.E)
        entframe.grid(row=1, sticky=tk.W+tk.E)

# Menubar
        menubar = tk.Menu(myframe)

        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="Run", command=self.run)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=sys.exit)
        menubar.add_cascade(label="File", menu=filemenu)
        myframe.config(menu=menubar)
        
# Toolbar

        try:
            okfig = tk.PhotoImage(file='ed_execute.gif')
            qtfig = tk.PhotoImage(file='bld_exit.gif')
            btok = tk.Button(toolframe, command=self.run, image=okfig)
            btqt = tk.Button(toolframe, command=sys.exit, image=qtfig)
            btok.pack(side=tk.LEFT)
            btqt.pack(side=tk.LEFT)
        except:
            print('gifs not found')

# Entries

        enwid = 11

        self.geo_d = {}
        nrow=0
        for key, val in geo_init.iteritems():
            lbl = tk.Label(entframe, text=key)
            var = tk.Entry(entframe, width=enwid)
            var.insert(0, val)
            self.geo_d[key] = var
            lbl.grid(row=nrow, column=0, sticky=tk.W+tk.E)
            var.grid(row=nrow, column=1, sticky=tk.W+tk.E)
            nrow += 1
        myframe.mainloop()


    def run(self):

        geo = {}
        for key, val in self.geo_d.iteritems():
            geo[key] = float(val.get())
        los_d = {}
        los_d['x0']    = -geo['y_det']
        los_d['y0']    = 0.001
        los_d['z0']    = geo['z_det']
        los_d['xend']  = geo['y_det']
        los_d['theta'] = np.radians(geo['tilt'])
        los_d['phi']   = -np.arctan(geo['tan_radius']/geo['y_det'])

        ctilt = np.cos(los_d['theta'])
        dy = geo['disk_thick']*ctilt
        ndisks = int(-2*geo['y_det']/dy)

        det_los = los.PHILOS(los_d, npoints=ndisks)

# Get separatrix {R,z}

        rmin = geo['Rmaj']  - geo['r_chamb'] 
        rmax = geo['Rmaj']  + geo['r_chamb'] 

# Restrict to LOS inside the plasma

        dr = det_los.rline[1] - det_los.rline[0]
        dz = det_los.zline[1] - det_los.zline[0]
        dl = np.hypot(dr, dz)
        ind = (det_los.rline > rmin - dl) & (det_los.rline < rmax + dl)
        r_in = det_los.rline[ind]
        z_in = det_los.zline[ind]
        x_in = det_los.xline[ind]
        y_in = det_los.yline[ind]

# Write ASCII file for LOS, used in the spectrum evaluation

        print('Writing ASCII output')
        y_los = x_in
        z_los = z_in
        det_pos = (geo['tan_radius'], geo['y_det'], geo['z_det'])
        det_dist = np.hypot(y_los - det_pos[1], z_los - det_pos[2])
        ctilt = np.cos(np.radians(geo['tilt']))
        stilt = np.sin(np.radians(geo['tilt']))
        dy = geo['disk_thick']*ctilt
        disk_radius = det_dist*np.tan(np.radians(geo['cone_aper']))
        n_disks = len(y_los)

# Each disk is divided in n_circles circular sectors,
# n_circles depends on the disk radius (which is almost constant)
# Every circular sector is divided in n_sectors sectors,
# equidistant poloidally; n_sectors is proportional to the radius
# of the circular sector

        out = '/ LOS\n' + \
              '/    y1 = %9.4f m\n'  % y_los[0]  + \
              '/    y2 = %9.4f m\n'  % y_los[-1] + \
              '/    z1 = %9.4f m\n'  % z_los[0]  + \
              '/    z2 = %9.4f m\n'  % z_los[-1] + \
              '/ Detector:\n' + \
              '/    Position [m]: x= %9.4f y=%9.4f z=%9.4f\n' %det_pos + \
              '/    Radius = %9.4f m\n' % geo['det_radius'] + \
              '/    Collimation angle = %9.4f deg\n' % geo['cone_aper'] + \
              '/ Disks:\n' + \
              '/    Thickness = %9.4f m\n' % dy + \
              '/    # disks = %5d\n' % n_disks + \
              '/ Cells:\n' + \
              '/    Radius = %9.4f m\n' % geo['cell_radius'] + \
              '/    # cells = xyz\n' + \
              '/    (x,y,x) cell cartensian coordinates [m]\n' + \
              '/    C = omega*Vol/4*pi = S*r/3 [m**3] is the volume in the solid angle\n' + \
              '/    Vol = cell volume [m**3]\n' + \
              '/    (u,v,w) LOS versor, cartensian coordinates\n' + \
              '/ \n' + \
              '/ x             y             z             C         ' + \
              '    Vol         u             v             w\n'

        n_cells=0
        cell_vec = np.zeros(3)
        for jdisk in range(n_disks):
            n_circles = int(0.5 + disk_radius[jdisk]/geo['cell_radius'])
            delta_radius = disk_radius[jdisk]/float(n_circles)

# radius, alpha in the 'middle' of the sector
# The central circle has only one sector (cell)
            cell_pos = (det_pos[0], -y_los[jdisk], z_los[jdisk])
            cell_los = (0., ctilt, stilt)
            cell_omega = geo['det_radius']**2 /det_dist[jdisk]**2 # *PI
            cell_vol  = np.pi * delta_radius**2 * dy
            cell_c    = cell_omega*cell_vol/4. #/PI
            out += ' %13.6E'*8 % (cell_pos + (cell_c, cell_vol) + cell_los) + '\n'
            n_cells += 1
            vec1 = - det_pos[1] - y_los[jdisk]
            vec2 = - det_pos[2] + z_los[jdisk]
            for j_circle in range(1, n_circles):
                n_sectors = 2*j_circle - 1
                radius = (j_circle + 0.5)*delta_radius
                cell_det_dist = np.hypot(det_dist[jdisk], radius)
                cell_omega = geo['det_radius']**2 * cell_det_dist/det_dist[jdisk]**3 #PI
                cell_vol = np.pi * delta_radius**2 * dy
                cell_c   = cell_omega*cell_vol/4. #/PI
                n_cells += n_sectors
                alpha = 0.5 + np.arange(n_sectors)
                rcos = radius*np.cos(alpha)
                rsin = radius*np.sin(alpha)
                for j_sector in range(n_sectors):
# LOS versor: cell position in the detector frame, normalised to one
                    cell_vec[0] = rcos[j_sector]
                    cell_vec[1] = vec1 - rsin[j_sector]*stilt
                    cell_vec[2] = vec2 + rsin[j_sector]*ctilt
                    cell_los = tuple(cell_vec/np.linalg.norm(cell_vec))
# cell_pos: with respect to torus center
                    cell_pos = tuple(np.array(det_pos) + cell_vec)
                    out += ' %13.6E'*8 % (cell_pos + (cell_c, cell_vol) + cell_los) + '\n'

        out = out.replace('xyz', str(n_cells))

        los_file = '%s.los' %tok_lbl
        flos = open(los_file, 'w')
        flos.write(out)
        flos.close()
        print('Written output file %s' %los_file)
        plot_los_cone()


if __name__ == '__main__':


    DET_LOS()
