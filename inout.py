import os
import numpy as np
from numpy.linalg import norm as norm
from ase.io.trajectory import TrajectoryReader
import matplotlib.pyplot as plt
import matplotlib
#from mpl_toolkits.mplot3d import Axes3D

########
#TODO: 
# - introduce numpy functions for reading/writing
# - update figure number handling

def read_xyz(filein, what='coor'):
    with open(filein, 'r') as f:
        Nat = int(f.readline())
        lines = f.readlines()
    species = np.empty(Nat, dtype='object')
    coor = np.zeros((Nat,3))
    for l in range(Nat):
        line = lines[l+1].strip().split()
        species[l] = line[0]
        coor[l] = np.array(line[1:]).astype(float)
    return species, coor

def read_trajxyz(filein, what='xqv'):
    with open(filein, 'r') as f:
        Nat = int(f.readline())
    if what=='x':
        os.system('awk -F " " \'{print $2, $3, $4}\' '+filein+' > trajtmp') 
        nc = 3
    else:
        os.system('awk -F " " \'{print $2, $3, $4, $5, $6, $7, $8}\' '+filein+' > trajtmp')
        nc = 7
    with open('trajtmp', 'r') as f:
        lines = f.readlines()
    os.system('rm trajtmp')
    Nt = int(len(lines)/(Nat+2))
    for t in range(Nt)[::-1]:
        del lines[t*(Nat+2):t*(Nat+2)+2]
    arr = np.zeros((len(lines), nc))
    for i in range(len(lines)):
        arr[i] = np.array(lines[i].strip().split(), dtype=np.float)
    arr = arr.reshape(Nt,Nat,nc)
    if what=='x':
        x = arr[:,:]
    else:
        x, q, v = arr[:,:,:3], arr[:,:,3], arr[:,:,4:7]

    if what == 'x':
        return x
    elif what == 'v':
        return v
    else:
        return x, q, v

def traj2xyz(filein, fileout):
    traj = TrajectoryReader(filein)
    Nat=len(traj[0].get_positions())
    with open(fileout, 'w') as f:
        for t in range(len(traj)):
            pos = traj[t].get_positions()
            print(Nat, file=f)
            print(' ', file=f)
            for a in range(Nat):
                print('C ', pos[a][0], ' ', pos[a][1], ' ', pos[a][2], file=f)

def read_optsteps(filein, Nat):
    with open( filein, 'r') as f:
        all_lines = (line.strip() for line in f)
        lines = list(line for line in all_lines if line)

    Nit = int(len(lines)/Nat)
    steps = []
    nline = 0
    for it in range(Nit):
        pos = []
        for i in range(Nat):
            nline = it*Nat + i
            line = lines[nline]
            ns = line.strip()
            posline = ns.split()
            at = [float(j) for j in posline]
            pos.append(at)
        steps.append(pos)
    return np.asarray(steps)


def plot_3Dstruct(structs, labels=[], showplane=True, view_ang=[22, -71], 
                            out='fig_struct.pdf'):

    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.view_init(view_ang[0], view_ang[1])

    if isinstance(structs[0,0], float): structs = np.array([structs])

    if showplane:
        xM, xm = np.amax(structs[0,:,0]), np.amin(structs[0,:,0])
        yM, ym = np.amax(structs[0,:,1]), np.amin(structs[0,:,1])
        X, Y = np.arange(xm-1, xM+1, 0.01), np.arange(ym-1, yM+1, 0.01)
        X, Y = np.meshgrid(X, Y)
        Z = 0*(X+Y)
        ax.plot_surface(X, Y, Z*0., color = '#2222dd', alpha = 0.1)

    for pos in structs:
        ax.scatter(pos[:,0], pos[:,1], pos[:,2], marker='o', depthshade=True)
        ax.plot(pos[:,0], pos[:,1], pos[:,2])

    ax.set_xlabel('x [ang]')
    ax.set_ylabel('y [ang]')
    ax.set_zlabel('z [ang]')
#    ax.legend()
    if out !=None:
        plt.savefig(out)
#    plt.figure(plt.gcf().number+1)

def plot_opt_evol(structs, calc, model='MBD',
                    outs=['fig_optevol_pol.pdf', 'fig_optevol_energy.pdf', 
                    'fig_optevol_dists.pdf']):

    Nat = len(structs[0,:,0])

    B = np.asarray([calc.calculate(pos, 'MBD', what='polmat', polmat_shape='tensor') 
                        for pos in structs])

    loc_i = B.sum(axis=4)
    tot_ab = loc_i.sum(axis=3)
    tot = np.trace(tot_ab, axis1=1, axis2=2)
    num = ( (tot_ab[:,0,0]-tot_ab[:,1,1])**2 
            + (tot_ab[:,0,0]-tot_ab[:,2,2])**2 
            + (tot_ab[:,1,1]-tot_ab[:,2,2])**2 )
    den = np.trace(tot_ab**2, axis1=1, axis2=2)
    ani = np.sqrt(num/den/2.)

    from molecular_bonds import get_H0_chain
    Embd, Ets, E0 = np.zeros(len(structs)), np.zeros(len(structs)), np.zeros(len(structs))
#    E0 = get_H0_chain(structs[0])
#    print(E0)
    E0 = np.asarray([get_H0_chain(pos) for pos in structs])
    Ets = np.asarray([calc.calculate(pos, 'TS', what='energy') for pos in structs])
    Embd = np.asarray([calc.calculate(pos, 'MBD', what='energy') for pos in structs])

    step = np.linspace(0, len(structs), num=len(structs))

    ## polarisation
    plt.figure()
    plt.xlabel('optimisation step')
    plt.gca().set_ylabel(r'$\alpha_{aa}$')
    plt.plot(step, tot, label = 'tot')
#    plt.plot(step, ani, label = 'ani')
    dirs = {0:'x', 1:'y', 2:'z'}
    for b in range(3):
        plt.plot(step, tot_ab[:,b,b], label=str(dirs[b]))
#    plt.plot(step, ani, marker='o')
    plt.legend()
    plt.savefig(outs[0])

    ## energy
    plt.figure()
    plt.scatter(step[10:], Embd[10:]-Embd[10], label= 'MBD', s=0.5)
    plt.scatter(step[10:], Ets[10:]-Ets[10], label='TS', s=0.5)
    plt.scatter(step[10:], E0[10:]-E0[10], label='bonds', s=0.5)
    plt.axhline(0, color='k', ls=':')
    plt.xlabel('optimisation step')
    plt.ylabel('$E(step)-E(input)$ [eV]')
    plt.legend(markerscale=10)
    plt.savefig(outs[1])

    ## dists
    plt.figure()
    z = structs[:,:,2].sum(axis=1)/Nat
    varz = np.sqrt(((z[:,None]-structs[:,:,2])**2).sum(axis=1))/Nat
#    plt.scatter(step[10:], z[10:], label = 'z', s=0.5)
    plt.scatter(step[10:], varz[10:], label = 'z coordinates', s=0.5)

   # avD = np.zeros(len(structs))
    d = np.zeros((len(structs), Nat-1))
    for it in range(len(structs)):
        pos = structs[it]
        for a in range(Nat-1):
            d[it,a] = norm(pos[a]-pos[a+1])
    avD = d.sum(axis=1)/Nat
#        avD[it] = np.sum([ norm(pos[i]-pos[i+1]) for i in range(Nat-1)])/Nat
    varD = np.sqrt(((avD[:,None]-d[:,:])**2).sum(axis=1))/Nat
#    plt.scatter(step[10:], avD[10:], label='inter-atom', s=0.5)
    plt.scatter(step[10:], varD[10:], label='inter-atom distance', s=0.5)
    plt.xlabel('optimisation step')
    plt.ylabel('variance')
    plt.legend()
    plt.savefig(outs[2])

def get_polmat_plot(pos, calc, label = '', out='fig_polmat.pdf'):

    plt.figure()
    Nat = len(pos)
    B = calc.calculate(pos, 'MBD', what='polmat')
    E = calc.calculate(pos, 'MBD', what='energy')
    odiag = B[~np.eye(B.shape[0],dtype=bool)].reshape(B.shape[0],-1)
    m, M = np.amin(odiag), np.amax(odiag)
    mval = max(-m, M)
    import matplotlib.colors as colors
    cnorm = colors.TwoSlopeNorm(0.0, vmin=m, vmax=M)
    p = plt.imshow(odiag, cmap='RdBu_r', norm=cnorm)
    plt.xticks([Nat/2, 3*Nat/2, 5*Nat/2], labels=['x', 'y', 'z'])
    plt.yticks([Nat/2, 3*Nat/2, 5*Nat/2], labels=['x', 'y', 'z'])
    plt.title(label + ', E=' + str(format('%.2f' % E)) + ' eV')
    
#                  yticks=[Nat/2, 3*Nat/2, 5*Nat/2], 
#                  yticklabels=['x', 'y', 'z'])
    plt.colorbar()
    plt.savefig(out)

def plot_polmat(B, label = '', out='fig_polmat.pdf', axes=False, scale='lin', cbloc='left', nodiag=True):

    Nat = int(len(B)/3)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if nodiag==True:
        mat = B[~np.eye(B.shape[0],dtype=bool)].reshape(B.shape[0],-1)
    else:
        mat = B
    m, M = np.amin(mat), np.amax(mat)
    mval = max(-m, M)
    import matplotlib.colors as colors
    if scale=='lin':
        cnorm = colors.TwoSlopeNorm(0.0, vmin=m, vmax=M)
    elif scale=='log':
        cnorm = colors.SymLogNorm(0.01, vmin=m, vmax=M, base=10)
    p = plt.imshow(mat, cmap='RdBu_r', norm=cnorm)
    if axes:
        for i in [1,2]:
            plt.axhline(i*Nat+0.5, color='#333333', linewidth=0.2, linestyle=':')
            plt.axvline(i*Nat+0.5, color='#333333', linewidth=0.2, linestyle=':')
    plt.xticks([Nat/2, 3*Nat/2, 5*Nat/2], labels=['x', 'y', 'z'])
    plt.yticks([Nat/2, 3*Nat/2, 5*Nat/2], labels=['x', 'y', 'z'])
    plt.title(label)
    cb = plt.colorbar(p, ax=[ax], location='left', label=r'$\alpha\ [a.u.]$')
    plt.savefig(out)

def plot_1Dproj(pos, calc, label='', out='fig_1Dproj.pdf', scale='linear'):

    Nat = len(pos)

    B = calc.calculate(pos, 'MBD', what='polmat')

    r, Br = [], []
    for i in range(Nat):
#        r.append(0.0)
#        Br.append(np.sum([B[i+a*Nat,i+a*Nat] for a in range(3)]))
        for j in range(i+1, Nat):
            r.append(norm(pos[i,:] - pos[j,:]))
            Br.append(np.sum([B[i+a*Nat,j+a*Nat] for a in range(3)]))
    ids = np.argsort(r)
    r = np.array(r)
    Br = np.array(Br)

    plt.figure()
    plt.xlabel("|r-r'| [ang]")
    plt.yscale(scale)
    if scale == 'linear':
        plt.ylabel("α (|r-r'|)")
    elif scale == 'log':
        plt.ylabel("log |α| (|r-r'|)")
        Br = np.abs(Br)
    plt.axhline(y=0, ls='--',color='k', linewidth=.3)
    plt.title(label)
    plt.plot(r[ids], Br[ids], linewidth=.5)
    plt.savefig(out)

def plot_1Dproj_hist1D(pos, calc, label='', out='fig_1DprojHist1D.pdf', fitdecay=False,
                            fitall=False):

    Nat = len(pos)

    B = calc.calculate(pos, 'MBD', what='polmat')

    r, Br = [], []
    for i in range(Nat):
#        r.append(0.0)
#        Br.append(np.sum([B[i+a*Nat,i+a*Nat] for a in range(3)]))
        for j in range(Nat):
            if i!=j:
                r.append(norm(pos[i,:] - pos[j,:]))
                Br.append(np.sum([B[i+a*Nat,j+a*Nat] for a in range(3)]))
    ids = np.argsort(r)
    r = np.array(r)
    Br = np.array(Br)

    Nbins = 100 # 200
    rmin = np.amin(r) ; rmax = np.amax(r)
    d = (rmax - rmin)/(Nbins-1)
    bins = np.linspace(0, Nbins-1, num=Nbins)
    binpos = rmin + bins*d + 0.5*d
    Bbin = np.zeros(Nbins)
    av = 0. ; last = 0 ; nl = 0
    for n in range(Nbins-1):
        lim = rmin + (n+1)*d
        for p in range(last, len(r)):
            ri = r[ids[p]]
            if ri-lim>1e-6: 
                Bbin[n]=av/(p-last+1e-6)
                av = 0. ; last = p
                break
            else:
                av+=Br[ids[p]]
                if n==Nbins-2: nl+=1
    Bbin[-1] = av/nl
    distplot = binpos[Bbin!=0]
    Bplot = np.abs(Bbin[Bbin!=0])

    plt.figure()
        
    plt.plot(distplot, Bplot)
    plt.axhline(0.0, color='k', ls=':')
    plt.xlabel("|r-r'| [ang]")
#    plt.yscale('log')
    plt.ylabel("α (|r-r'|)")
    cut=15.
    if fitdecay:
        def f(x, cost, gamma):
            return cost + x**(-gamma)
        from scipy.optimize import curve_fit as fit
        popt, pcov = fit(f, distplot[distplot>cut], Bplot[distplot>cut])
        plt.plot(distplot[distplot>cut], f(distplot[distplot>cut], popt[0], popt[1]),
                    label = "|r-r'|^(-"+ str(format('%.4f'%popt[1]))+' \pm ' +
                        str(format('%.4f'%pcov[1,1]))+')', ls='--', lw=1)
    if fitall:
        cut =5.
        def f(x, cost, gamma):
            return cost - x**(-gamma) #+ x**(-1.51)
        from scipy.optimize import curve_fit as fit
        popt, pcov = fit(f, distplot[distplot<cut], Bplot[distplot<cut], [0., 6])
        plt.plot(distplot[distplot<cut], f(distplot[distplot<cut], popt[0], popt[1]),
                    label = str(format('%.2f'%popt[0])) + " + |r-r'|^(-" 
                    + str(format('%.2f'%popt[1]))+')', ls='--', lw=1)

    plt.legend()
    plt.savefig(out)

def plot_1Dproj_hist(pos, calc, label='', out='fig_1DprojHist.pdf', scale='linear'):

    Nat = len(pos)

    B = calc.calculate(pos, 'MBD', what='polmat')

    r, Br = [], []
    for i in range(Nat):
 #       r.append(0.0)
 #       Br.append(np.sum([B[i+a*Nat,i+a*Nat] for a in range(3)]))
        for j in range(Nat):
            if i!=j:
                r.append(norm(pos[i,:] - pos[j,:]))
                Br.append(np.sum([B[i+a*Nat,j+a*Nat] for a in range(3)]))
    ids = np.argsort(r)
    r = np.array(r)
    Br = np.array(Br)
    plt.figure()
    plt.axhline(0.0, color='k', ls='--')
    h = plt.hist2d(r, Br, bins=150, cmap='hot', rasterized=True)
#    h = plt.hist2d(r[ids], np.log(np.abs(Br[ids])), bins=150, cmap='viridis', rasterized=True, norm=matplotlib.colors.LogNorm())
    plt.xlabel("|r-r'| [ang]")
    plt.ylabel("α (|r-r'|)")
    plt.title(label)
    plt.colorbar(h[3])
    plt.savefig(out, dpi=300)

def plot_polmat_group(structs, calc, labels, 
                        view_ang=[22,-78], nhfigs=2, outs=[
                        'opt_structs', 'opt_polmat', 'opt_poldiag',
                        'opt_1Dproj']):

    # polarization colorplots optimised for 4 plots

    N = len(structs)
    Nat = len(structs[0])
    fign = range(plt.gcf().number, plt.gcf().number+4)

    ## 1, structure plot
    fig1 = plt.figure(1)#fign[0])
    ax = fig1.add_subplot(111, projection='3d')
    ax.view_init(view_ang[0], view_ang[1])
    ax.set_xlabel('x [ang]')
    ax.set_ylabel('y [ang]')
    ax.set_zlabel('z [ang]')
    # xy axis coloring
    xM, xm = np.amax(structs[0,:,0]), np.amin(structs[0,:,0])
    yM, ym = np.amax(structs[0,:,1]), np.amin(structs[0,:,1])
    X, Y = np.arange(xm-1, xM+1, 0.01), np.arange(ym-1, yM+1, 0.01)
    X, Y = np.meshgrid(X, Y)
    Z = 0*(X+Y)
    ax.plot_surface(X, Y, Z*0., color = '#2222dd', alpha = 0.1)
#    plt.figure(plt.gcf().number+1)

    ## 2, polarizability maps
#    plt.figure(2)#fign[1])
    if nhfigs == 0:
        fig2, ax2 = plt.subplots(N)
    else:
        fig2, ax2 = plt.subplots(int(N/nhfigs),nhfigs)
    plt.subplots_adjust(wspace=0.7)
    plt.setp(ax2, xticks=[Nat/2, 3*Nat/2, 5*Nat/2], 
                  xticklabels=['x', 'y', 'z'],
                  yticks=[Nat/2, 3*Nat/2, 5*Nat/2], 
                  yticklabels=['x', 'y', 'z'])

    B = [ calc.calculate(pos, 'MBD', what='polmat') for pos in structs]
    diag = np.empty((0,3*Nat))
    odiag = np.empty((0,3*Nat, 3*Nat-1))
    for Bn in B[:]:
        diag_n = Bn.diagonal()
        odiag_n = Bn[~np.eye(Bn.shape[0],dtype=bool)].reshape(Bn.shape[0],-1)
        diag = np.append(diag, [diag_n], axis=0)
        odiag = np.append(odiag, [odiag_n], axis=0)
    B = np.asarray(B)

    M = np.amax(odiag)
    m = np.amin(odiag)

    cx=0; cy=0
#    plt.figure(plt.gcf().number+1)

    ## 3, trace
    fig3 = plt.figure(3)#fign[2])
    els = np.asarray(range(3*Nat))
    plt.xticks([Nat/2, 3*Nat/2, 5*Nat/2],['x', 'y', 'z'])
    fig3.gca().set_ylabel(r'$\alpha$ (r,r)')
#    plt.figure(plt.gcf().number+1)

    ## 4, 1d projection
    if nhfigs == 0:
        fig4, ax4 = plt.subplots(N)
    else:
        fig4, ax4 = plt.subplots(int(N/nhfigs),nhfigs)
    fig4.subplots_adjust(hspace=0.5, wspace=0.5)
#    plt.figure(plt.gcf().number+1)

    # structure processing

    for n in range(N):

        pos = structs[n]

        s=0.
        for i in range(len(B)):
            for j in range(len(B)):
                s+=B[n,i,j]
        
        E = calc.calculate(pos, 'MBD', 'energy')

        plt.figure(fign[0])
        plt.plot(pos[:,0], pos[:,1], pos[:,2], label=labels[n],marker='o',markersize=4)
        
        plt.figure(fign[1])
        if nhfigs == 0:
            panel = ax2[cy]
        else:
            panel = ax2[cx, cy]
        panel.set_title(labels[n]+', E='+str(format('%.2f' % E)))
        p = panel.imshow(odiag[n], cmap='inferno', vmin=m, vmax=M)
        fig2.colorbar(p, ax=panel)

        plt.figure(fign[2])
        plt.plot(els, diag[n], label=labels[n])

        plt.figure(fign[3])
        if nhfigs == 0:
            panel = ax4[cy]
        else:
            panel = ax4[cx, cy]
        r, Br = [], []
        # diagonal part inclusion???
        for i in range(Nat):
            r.append(0.0)
            Br.append(np.sum([B[n,i+a*Nat,i+a*Nat] for a in range(3)]))
            for j in range(i+1, Nat):
                r.append(norm(pos[i,:] - pos[j,:]))
                Br.append(np.sum([B[n,i+a*Nat,j+a*Nat] for a in range(3)]))
        ids = np.argsort(r)
        r = np.array(r)
        Br = np.array(Br)
        panel.set_xlabel("|r-r'| [ang]")
        panel.set_ylabel("α (|r-r'|)")
        panel.axhline(y=0, ls='--',color='k', linewidth=.3)
        panel.set_title(labels[n])
        panel.plot(r[ids], Br[ids], linewidth=.5)

        print(n, np.sum(diag[n]), np.sum(Br))

        cy+=1
        if cy == nhfigs:
            cy=0
            cx+=1

    ax.legend()
    fig3.legend()
#    fig4.legend()
    for fig in range(1,5):
        plt.figure(fig)
        plt.savefig('fig_'+outs[fig-1]+'.pdf')

def plot_polar_group(steps, outs=['fig_polar_angspan.pdf', 'fig_polar_2Dangs.pdf', 'fig_radialspan.pdf']):

    from structures import cartesian2polar, polar2cartesian
    import matplotlib.colors as colors

    pol = np.empty((0,3))
    for coor in steps[:]:
            poli = cartesian2polar(coor)
            pol = np.append(pol, poli, axis=0)

    Nbins = 500

    d = 2*np.pi/Nbins
    ang_map = np.zeros((Nbins,Nbins))
    ang0 = [-np.pi/2, -np.pi/2]

    for point in pol:
        angs_point = point[1:]
        idbins = (angs_point-ang0)/(d)
        x, y = np.around(idbins).astype(int)
        if x >= Nbins: x -= Nbins
        if y >= Nbins: y -= Nbins
        ang_map[x, y] +=1
    ang_map += 0.00001

    ticks = [np.linspace(0, int(Nbins/2.), 5), 
            np.linspace(0, Nbins, 5)]
    xlabels = []
    for a in range(2):
        xlabels.append(["{:.1f}".format( (ang0[a]+i*d)/np.pi) + '$\pi$' for i in ticks[a] ])

    phionly = [np.sum(ang_map[i,:]) for i in range(Nbins)]
    thetaonly = [np.sum(ang_map[:,i]) for i in range(Nbins)]

    plt.figure()
    plt.plot(np.arange(Nbins), thetaonly, label = 'theta')
    plt.plot(np.arange(Nbins), phionly, label = 'phi')
    plt.yscale('log')
    plt.gca().set_xticks(ticks[1])
    plt.gca().set_xticklabels(xlabels[1])
    plt.xlabel('angle')
    plt.ylabel('counts')
    plt.legend()
    plt.savefig(outs[0])

    plt.figure()
    plt.imshow(ang_map[:,:int(Nbins/2)], origin='lower', cmap='Blues', aspect='auto', norm=colors.LogNorm(vmin=0.1, vmax = np.amax(ang_map)), interpolation='bicubic')
    plt.gca().set_xticks(ticks[0])
    plt.gca().set_xticklabels(xlabels[0])
    plt.gca().set_yticks(ticks[1])
    plt.gca().set_yticklabels(xlabels[1])
    plt.gca().set_xlabel(r'$\theta$, xz plane')
    plt.gca().set_ylabel(r'$\phi$, xy plane')
    plt.colorbar()
    plt.savefig(outs[1])

    plt.figure()
    hist, dbins, dpatches = plt.hist(pol[:,0], 100, facecolor='blue', alpha=0.5)
    plt.axvline(1.52, lw=0.5, ls='--', color='k', label = 'harmonic bond equilibrium distance')
    plt.yscale('log')
    plt.xlabel('inter-atomic distance [ang]')
    plt.ylabel('counts')
    plt.legend()
    plt.savefig(outs[2])

'''
def read_struct(filein):
    with open(filein, 'r') as f:
        lines = np.asarray(f.readlines())
    pos = np.empty((len(lines),3))
    c=0
    for line in lines:
        ns = line.strip()
        at = np.array([float(j) for j in ns.split()])
        pos[c] = at
        c+=1
    return pos
'''
