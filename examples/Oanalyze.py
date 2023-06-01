from ase.io.trajectory import TrajectoryReader
from os import system
import numpy as np
import os.path
from scipy.linalg import eig as diag
from numpy.linalg import norm as norm
from scipy.fft import fft, rfft
import matplotlib.pyplot as plt
#from setup_systems import run_mbd, get_hessian
#from setup_systems import set_chain

from scipy.signal import argrelextrema

#import MDAnalysis as mda
#from MDAnalysis.coordinates.XYZ import XYZReader as reader

## temp forces modules
from ase.atoms import Atoms
from ase.calculators.harmonic import default_avg_a_div_a0
from ase.calculators.libmbd import MBD as MBDcalc
##

def traj2xyz(basename):
    traj = TrajectoryReader(basename + '.traj')
    Nat=len(traj[0].get_positions())
    with open(basename + '.xyz', 'w') as f:
        print(Nat, file=f)
        print(' ', file=f)
        for t in range(len(traj)):
            pos = traj[t].get_positions()
            for a in range(Nat):
                print('C ', pos[a][0], ' ', pos[a][1], ' ', pos[a][2], file=f)

def mode(N, k, A0, Ak):
    
    f = []
    
    for a in range(N):
        #f.append(2*Ak/np.sqrt(N)*np.exp(2*np.pi*1j*k*a/N))
#        f.append(-18.9/np.sqrt(N)*np.exp(2*np.pi*1j*k*a/N))
        f.append(-18.9*np.cos(np.pi*k*a/N + np.pi/2))

    return f

def get_av_struct(traj, tcut):

    start = traj[0].get_positions()
    Nat = len(start)
    ttot = len(traj)
#    ttot = 2000
    
    A = []
    avA = np.zeros(int(Nat/2+1))
    norm = ttot - tcut
    ch = 0
    for t in range(tcut, ttot):

        ch += 1
        f = traj[t].get_positions()
        x = f[:, 0]
        y = f[:, 1]
        At = rfft(np.asarray(y), norm='ortho')

        avA += np.real(At)
        A.append(np.real(At))
        
    avA = avA/norm
    errA = np.zeros(int(Nat/2)+1)

    for t in range(len(A)):
        errA += (avA-A[t])**2/(norm-1)       
    errA = np.sqrt(errA)

    return avA, errA

def get_forces_temp(pos):

    atoms = Atoms('C'+str(len(pos)), positions=(pos))
    calc = MBDcalc(mode='fortran', get_MBD_spectrum = False)
    hvr = np.array([default_avg_a_div_a0['C'],]*len(pos))
    calc.set_a_div_a0(hvr)
    calc.calculate(atoms=atoms)

    return calc.results['forces']

def compare_structs(traj, idx, title, fileout):

    Nat = len(traj[0].get_positions())

    f = traj[0].get_positions()

    fig, axs = plt.subplots(2)

    axs[0].set_title(title)

    colors = [ 'crimson', 'dodgerblue', 'forestgreen', 'gold']

    wv = [ k/Nat for k in range(1, int(Nat/2)+1) ]
    
    axs[1].set_xlabel('x [Ang]')
    axs[0].set_ylabel('y [Ang]')
    axs[0].plot(f[:,0],f[:,1], 'k--')

#    axs[1].set_ylabel('$F_x$')
#    axs[1].axhline(0., lw = .2, color='black')
    
    axs[1].set_ylabel('$F_y$')
    axs[1].yaxis.set_label_coords(-0.1,0.5)
    axs[1].axhline(0., lw = .2, color='black')
    l = 3
    dx = pos[1,0]-pos[0,0]
    xmin = pos[l,0] - dx/2.
    xmax = pos[len(pos)-l,0] + dx/2.
    axs[1].axvline(xmin, 0, 1)
    axs[1].axvline(xmax, 0, 1)

#    axs[3].set_xlabel('$\omega [rad/ps]$')
#    axs[3].set_ylabel('$A(k)$')
#    axs[3].axhline(0., lw = .2, color='black')

    for i in range(len(idx)):

        print(idx[i][0])
        f = traj[idx[i][0]].get_positions()
        x = f[:, 0]
        y = f[:, 1]
        A = rfft(np.asarray(y), norm='ortho')

        av_k = 0.
        for k in range(len(A)):
            av_k += k/Nat*abs(A[k])
        av_k = av_k/np.sum(abs(A))

        if len(idx[0]) == 2:
            lab = 't = ' + str(idx[i][0]) + ', count=' + str(idx[i][1])
        elif len(idx[0]) == 1:
            lab = 't = ' + str(idx[i][0]*2e0)+' fs'
        else:
            exit('idx[0] is not an array of lentgh 1 or 2')

#        axs[0].plot(x,y, label = lab, lw = 2., color=colors[i])
        axs[0].scatter(x,y, label = lab, lw = 2., color=colors[i])

    #    E, F = run_mbd(f, 'MBD', 'forces')
    ## temporary fix!!!
        F = get_forces_temp(f)
#        axs[1].plot(x, F[:,0], color = colors[i], ls = 'solid', label = 'x')
        tot = 0.
        for a in range(2,len(F[:,1])-2):
#            print(F[a,1], tot)
            tot += F[a,1]
#        tot = '{:.4f}'.format(tot/len(F[:,1]))
        tot = '{:.2e}'.format(tot/(len(F[:,1])-4))
        axs[1].plot(x, F[:,1], color = colors[i], ls = 'solid', label = 'tot='+tot)
        lab = 't = ' + str(idx[i][0]) + ', $k_{av}$=' + '{:.4f}'.format(av_k)
#        axs[3].scatter(wv,abs(A[1:]), facecolors ='none', edgecolors=colors[i], s=40, label = lab, lw = 2.)

    axs[0].legend(bbox_to_anchor=(.8,1.1),loc='upper left')
    axs[1].legend(bbox_to_anchor=(.75,0.25),loc='upper left')
#    axs[3].legend()
    if fileout != None:
        plt.savefig(fileout)
    plt.show()

def find_recurrent_structs(traj, tcut, Nstructs):

    ttot = len(traj)
    Nat = len(traj[0].get_positions())
    recurrent = []
    idx = []
    struct_count = []
    for t in range(tcut,ttot):
        f = traj[t].get_positions()
        A = rfft(np.asarray(f[:,1]), norm='ortho')
        norm = np.sum(abs(A))
        part = abs(A)/norm
        new = True
        for struct in range(len(recurrent)):
            diff = 0.
            for a in range(len(part)):
                if (part[a]>=1e-2):
                    diff += (part[a]-recurrent[struct][a])**2
            if (np.sqrt(diff) >= 1e-2):
                pass
            else:
                struct_count[struct] += 1
                new = False
                break
        if new:
            recurrent.append(part)
            idx.append(t)
            struct_count.append(1)
    
    struct_count = np.asarray(struct_count)
    top = struct_count.argsort()[-Nstructs:][::-1]
    id_top = []
    for n in top:
        id_top.append([idx[n], struct_count[n]])

    return id_top

def find_wv_av(traj):
    # structure corrugations

    ttot = len(traj)
    Nat = len(traj[0].get_positions())

    highpart = []
    wv = []
    for t in range(ttot):
       f = traj[t].get_positions()
       A = rfft(np.asarray(f[:,1]), norm='ortho')
       av_wv = 0.
       for k in range(len(A)):
           av_wv += a/Nat*abs(A[a])
       av_wv = av_wv/np.sum(abs(A))
       wv.append(av_w)

    wv = np.asarray(wv) + tcut*np.ones(len(wv))
    return  np.asarray(wv)

def plot_Avstime(traj, id0, idf, fileout):

    pos = traj[0].get_positions()
    Nat = len(pos)
    dt = .2
    tmax = len(traj)

    A0 = np.asarray(rfft(np.asarray(pos[:,1]), norm='ortho'))
    time = []
    A = []
    for t in range(1,len(traj)):
        pos = traj[t].get_positions()
        time.append(t*2e-3) # time in ps
        At = np.asarray(rfft(np.asarray(pos[:,1]), norm='ortho'))
        A.append(abs(At)/abs(A0))
    A = np.asarray(A)

    ## k = 1 reconstruction

#    A = abs(A)
    fig, axs = plt.subplots(4)
#    fig1 = plt.figure(1)
    axs[0].set_title('$|A_k(t)|/|A_k(0)|$: new chain, NVE ensamble with MBD')
    axs[3].set_xlabel('t [ps]')
    plt.setp([a.get_xticklabels() for a in axs[:-1]], visible=False)
    plt.subplots_adjust(wspace=0, hspace=0)

    for a in range(id0,idf):
        k = a/Nat
        axs[a-1].set_ylabel('k='+str(a))
        axs[a-1].plot(time, A[:, a], label = 'k = ' + str(k), lw = .5)
#        axs[a-1].legend()
        y = np.asarray(A[:,a])
        peaks = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]))[0] + 1
        for i in range(len(peaks)):
            axs[a-1].vlines(peaks[i]*2e-3, 0., 1.25, lw = .4, color='black')
    


    if (fileout != 'none'): plt.savefig(fileout)

    plt.show()

'''
#    print(np.shape(A[:,1]))
    Akw = np.asarray(rfft(np.asarray(A[:,1]), norm='ortho'))
    Nw = len(Akw)
#    wmin = 2*np.pi/tmax
#    wmax = 2*np.pi/dt
#    freq = np.arange(0.,wmax,wmin)
    # reconstruct A from F transform --> dominant freq. vs k

    rec = []
#    for t in range(200):
    for t in range(1,tmax):
        sum = 0.
        for w in range(Nw):
            sum+=Akw[w]/np.sqrt(Nw*2)*np.cos(2*np.pi*w*t/Nw)
        rec.append(np.real(sum))
    rec = np.asarray(rec)

    freq = [ w/Nw for w in range(Nw) ]
    freq = np.asarray(freq)
#    time = np.asarray([t for t in range(200)])
    time = np.asarray([t for t in range(1,tmax)])

    plt.plot(time, rec, label ='rec')
    plt.plot(time, A[:,1], label ='original')
    plt.legend()
    plt.show()
'''
def get_dispersion_relation(traj, ti, tf):

    pos = traj[0].get_positions()
    A0 = np.asarray(rfft(np.asarray(pos[:,1]), norm='ortho'))
    nw = len(A0)

    A = []
    ttot = 0.
    for t in range(ti,tf):
        pos = traj[t].get_positions()
#        time.append(t*2e-3) # time in ps
        ttot += 2e-3
        At = np.asarray(rfft(np.asarray(pos[:,1]), norm='ortho'))
        A.append(abs(At)/abs(A0))
    A = np.asarray(A)
    w = []
    for k in range(1,nw):
        y = np.asarray(A[:,k])
        peaks = np.where((y[1:-1] > y[0:-2]) * (y[1:-1] > y[2:]))[0] + 1
        T = ttot/len(peaks)
        w.append(2*np.pi/T)
    return np.asarray(w)


def plot_modes_deloc(traj, idx):
    # plot inverse participation ration (IPR)

    fig, axs = plt.subplots(2)

    axs[0].set_ylabel('$ \epsilon_\lambda$')

    axs[1].set_xlabel('MBD mode')
    axs[1].set_ylabel('IPR')

    for t in idx:

        pos = traj[t].get_positions()
        Nat = len(pos)

        E, evals, chi = run_mbd(pos, 'MBD', 'spectrum')

        ipr = []
        for mode in range(3*Nat):
            pl = 0.
            for a in range(Nat):
                norm = 0.
                for dir in range(3):
                    norm += chi[mode, 3*a + dir]**2
                pl += norm**2
            ipr.append(1/Nat/pl) 

        modes = np.asarray([ a/3/Nat for a in range(3*Nat)])

        axs[0].plot(modes, evals, label='t = ' + str(t*2e-3) + ' ps')
        axs[1].plot(modes, ipr)

    axs[0].legend()
    plt.show()

def get_vibrational_modes(pos, model):

    model = 'MBD'

    hessian = get_hessian(pos, model)

    evals, evecs = diag(hessian)

    print(evals[:])

def distancesVtime(traj):

    time = []
    L = []
    avd = []
    pos = traj[0].get_positions()
    L0 = norm(pos[-1]-pos[0])
    d0 = 0.
    for a in range(1,len(pos)):
        d0 += norm(pos[a]-pos[a-1])/len(pos)

    for t in range(1,int(1e4)):
        pos = traj[t].get_positions()
        time.append(t*2e-3)
        d = 0.
        for a in range(1,len(pos)):
            d += norm(pos[a]-pos[a-1])
        avd.append(d/len(pos)/d0)
        L.append(norm(pos[-1]-pos[0])/L0)

    return np.asarray(time), np.asarray(avd), np.asarray(L)

def momenta_distribution(traj, dir, acut):
    N = len(traj[0].get_positions())
    pdir = []
    for t in range(len(traj)):
        p = np.asarray(traj[t].get_momenta())
        pdir.append(p[acut:N-acut,dir])
    pdir = np.asarray(pdir)
    pdir = pdir.flatten()

    return pdir

############################

#tcut = 500

## laptop
#traj = TrajectoryReader('./data/chain70MBD.traj')

## desktop
#traj = TrajectoryReader('./nve/chain.traj')
#traj = TrajectoryReader('./50mbd/chain1e4.traj')


#######

## early traj study
#pos = traj[0].get_positions()
#N = len(pos)
#print(pos)
#print('vibs')
#get_vibrational_modes(pos, 'MBD')

## transversal momenta distribution

#nvt = TrajectoryReader('./nve/chain.traj')
#nve = TrajectoryReader('./50mbd/chain1e4.traj') #spaced
#nve = TrajectoryReader('./50mbd/chain.traj')

#plot_Avstime(mbd, 1, 5, 'none')

## dispersion relation
'''
mbd = TrajectoryReader('./nvt1mbd/nodes.traj')
ts = TrajectoryReader('./nvt1ts/nodes.traj')
nvembd = TrajectoryReader('./nvembd/nodes.traj')
nvets = TrajectoryReader('./nvets/nodes.traj')

N = len(mbd[0].get_positions())

wmbd = get_dispersion_relation(mbd, 0, 1000)
wts = get_dispersion_relation(ts, 0, 1000)
wnvembd = get_dispersion_relation(nvembd, 0, 1000)
wnvets = get_dispersion_relation(nvets, 0, 1000)

k = np.linspace(0, len(wmbd)*np.pi/(N-1), num=len(wmbd))

plt.plot(k, wmbd, label ='nvt: mbd, 1 K')
plt.plot(k, wts, label ='nvt:  ts, 1 K')
plt.plot(k, wnvembd, label ='nve: mbd', lw = 6.)
plt.plot(k, wnvembd, label ='nve: ts ')
plt.legend()
plt.show()
'''
'''
Nat = 40
pos = []
#with open( 'MBD3D40_n5_f2.0tol3.txt', 'r') as f:
with open( 'gs3D40_n5_f2.0tol5scratch.txt', 'r') as f:
    for i in range(Nat):
        line = f.readline()
        ns = line.strip()
        posline = ns.split()
        at = [float(j) for j in posline]
        pos.append(at)
pos = np.asarray(pos)
get_pol_matrix(pos)
'''
'''

tcut = min(len(nvt),len(nve))

tcut = 10000

px_nve = momenta_distribution(nve[:tcut], 0, 4)
px_nvt = momenta_distribution(nvt[:tcut], 0, 4)

py_nve = momenta_distribution(nve[:tcut], 1, 4)
py_nvt = momenta_distribution(nvt[:tcut], 1, 4)



figs, axs = plt.subplots(2, 2)

xmin = np.amin(np.concatenate((px_nve, px_nvt)))
xmax = np.amax(np.concatenate((px_nve, px_nvt)))
xlim = (xmin, xmax)
plt.setp([axs[1][0].set_xticks(np.arange(xmin*1.5, xmax*1.5, step = (xmax-xmin)/5)) for a in range(2)])

xmin = np.amin(np.concatenate((py_nve, py_nvt)))
xmax = np.amax(np.concatenate((py_nve, py_nvt)))
xlim = (xmin, xmax)
plt.setp([axs[1][1].set_xticks(np.arange(xmin*1.5, xmax*1.5, step = (xmax-xmin)/5)) for a in range(2)])
xmin = np.amin(np.concatenate((py_nve, py_nvt)))
xmax = np.amax(np.concatenate((py_nve, py_nvt)))
'''
'''
ylim = (-100, 6500)
#ylim = (-0.01, 1.1)
plt.setp([axs[a][0] for a in range(2)], xlim=(-12.5, 12.5), ylim=ylim)
plt.setp([axs[a][1] for a in range(2)], xlim=(-20, 20), ylim=ylim)

plt.setp([axs[0][a].get_xticklabels() for a in range(2)], visible = False)

plt.setp([axs[a][1].get_yticklabels() for a in range(2)], visible = False)
#plt.setp([axs[a][0].set_yticks(np.arange(0,6000, step=1000)) for a in range(2)])

figs.suptitle('Momentum distribution: NVE vs NVT')

axs[0,0].set_ylabel('counts')
axs[1,0].set_ylabel('counts')
axs[1,0].set_xlabel('$p_x$')
axs[1,1].set_xlabel('$p_y$')

[[axs[a,b].axhline(0., lw = .2, color='black') for a in range(2)] for b in range(2)]
#axs[1,0].axhline(0., lw = .2, color='black')


plt.setp([a.get_xticklabels() for a in axs[:][0]], visible=False)
'''

# fit and normalisation attempt

'''
n, bins, patches = plt.hist(px_nve, bins='auto', label = 'nve, x')#, visible=False)
#plt.ylim(-0.01, 0.1)
tot = np.sum(n)
plt.plot(bins[1:], n)#/tot)
plt.plot(bins[:-1], n)#/tot)
half = []
for a in range(len(bins)-1):
    half.append((n[a]-bins[a-1])/2.)
plt.plot(np.asarray(half), n)
plt.show()
from statistics import NormalDist
norm = NormalDist.from_samples(px_nve)
print(norm.mean, norm.stdev)
#axs[0,0].plot()
axs[0,0].hist(px_nve, bins='auto', label = 'nve, horizontal')
axs[1,0].hist(px_nvt, bins='auto', label = 'nvt, horizontal')
axs[0,1].hist(py_nve, bins='auto', label = 'nve, vertical')
axs[1,1].hist(py_nvt, bins='auto', label = 'nvt, vertical')

[[axs[a,b].legend() for a in range(2)] for b in range(2)]
plt.subplots_adjust(wspace=0, hspace=0)
plt.savefig('momenta_NVEvsNVT.pdf')
plt.show()
'''


#px_nve = momenta_distribution(nve[:tcut], 0, 4)

#from pylab import *
#from scipy.optimize import leastsq

#func = lambda p, x: p[0]*exp(-0.5*((x-p[1])/p[2])**2)+p[3]
#err = lambda p, x, y: (y - func(p, x))


##### mbd debug
'''
pos, posH, posS = set_chain(3, 1.52, 'straight', 1, 2)
ang= 1.889726
pos_ang = [[j*ang for j in i] for i in pos]
'''
#run_mbd(pos, 'MBD', 'none')
#compare_structs(traj, [[40], [7302], [1810]], '50K, MBD', 'Fy_50mbd_sample.pdf')
#compare_structs(traj, [[40], [165], [395]], 'nve', 'Fy_nve_sample.pdf')

## momenta distribution
#figs, axs = plt.subplots(3)

#momenta_distribution(traj, 10, 3)
#p = traj[15].get_momenta()
#data = p[3:N-3,1]
#print(data)
'''
dbin = max(data)-min(data)/15
bins = np.arange(min(data), max(data), dbin)
plt.xlim([min(data)-dbin*1.5, max(data)+dbin*1.5])

plt.hist(data, bins=bins, alpha=0.5)
'''
#print(np.sort(data))

#plt.show()

## inter-atomic distances v time: nvt vs nve
'''
traj_nvt = TrajectoryReader('./50mbd/chain.traj')
traj_nve = TrajectoryReader('./nve/chain.traj')

t_nvt, avd_nvt, L_nvt = distancesVtime(traj_nvt)
t_nve, avd_nve, L_nve = distancesVtime(traj_nve)

fig, axs = plt.subplots(2)
axs[0].set_title('Average interatomic distance over time')
axs[0].set_ylabel('d(t)/d(0)')
axs[1].set_ylabel('d(t)/d(0)')
axs[1].set_xlabel('time [ps]')
axs[0].plot(t_nve, avd_nvt, label = 'nvt', lw=.3)
axs[1].plot(t_nve, avd_nve, label = 'nve', lw=.3)
#plt.plot(t_nvt, L_nvt, label = 'L, nvt')
#plt.plot(t_nve, L_nve, label = 'L, nve')

#print('nvt:', avd[-1]


axs[0].legend()
axs[1].legend()
plt.show()
'''

## Avtime

#plot_Avstime(traj, 1, 5, 'Avtime_nve.pdf')
#plot_Avstime(traj, 1, 5, 'Avtime_50mbd.pdf')

#plot_Avstime(traj, 1, 5, 'Avtime_spacedNVE.pdf')

## average Ak: NVT vs NVE

'''
traj_nve = TrajectoryReader('./nve/chain.traj')
traj_nvt = TrajectoryReader('./50mbd/chain1e4.traj')
avE, avErrE = get_av_struct(traj_nve,0)
avT, avErrT = get_av_struct(traj_nvt,0)
Nat = 40
w = [ k for k in range(1,int(Nat/2)) ]
msize = 5
plt.title('Time-averaged |$A_k$| for <R>/$R_0$=1 with MBD')
plt.errorbar(w, abs(avE[1:int(Nat/2)]), yerr = avErrE[1:int(Nat/2)], fmt = 'b.', markersize = msize, label = 'NVE')
plt.errorbar(w, abs(avT[1:int(Nat/2)]), yerr = avErrT[1:int(Nat/2)], fmt = 'r.', markersize = msize, label = 'NVT, 50 K')
plt.axhline(0., lw = .2, color='black')
plt.ylabel('$|<A_k>|$')
plt.xlabel('$k$')
#plt.savefig('avA-grad70.pdf')
#print(Ambd)
#print(Ats)
plt.savefig('averageAk_spaced.pdf')
plt.show()
'''

## regular vs irregular structs
'''
w = find_avfreqs(traj)
t_wmin, t_wmax = w.argsort()[-1], w.argsort()[0]
print(t_wmin, t_wmax)
compare_structs(traj, [[7302], [1810]], 'Low vs high frequency: chainMBD_grad70', None)
'''

## frequent modes
'''
id_modes = find_recurrent_structs(traj, tcut, 4)
compare_structs(traj, id_modes, 'chainTS_grad70', None)
'''

## delocalisastion of MBD
'''
idx = [7302, 1810]
plot_modes_deloc(traj, idx)
'''

## vibrational modes for a structure
'''
get_vibrational_modes(traj[570].get_positions(), 'MBD')
plot_Avstime(traj, 1, 5, 'dominantAvtime-chainTSgrad70.pdf')
'''

## 'dispersion' relation
'''
from scipy.optimize import curve_fit

pos = traj[0].get_positions()
Nat = len(pos)
A0 = np.asarray(rfft(np.asarray(pos[:,1]), norm='ortho'))
time = []
A = []
for t in range(1000,len(traj)):
    pos = traj[t].get_positions()
    time.append(t*2e-3) # time in ps
    At = np.asarray(rfft(np.asarray(pos[:,1]), norm='ortho'))
    A.append(abs(At)/abs(A0))

A = abs(np.asarray(A))

# cosine/sine fit attempt

def func(t, cost, amp, w, phi):
#    print(type(t), type(amp), type(w), type(phi))
    return cost + amp*np.sin(w*t + phi)

#popt, popcov = curve_fit(func, time, A[:,1], p0 = [0.84, 0.1, 15.0, -1.5])

# Fourier transform attempto

fAt = rfft(A[:,1], norm='ortho')
dt = time[1]-time[0]
print(shape(fAt))


plt.plot(time, func(np.asarray(time), popt[0], popt[1], popt[2], popt[3]), 'r-')
print(popt)
plt.plot(time, func(np.asarray(time), 0.84, 0.1, 15.16479, -1.44))
plt.plot(time, A[:,1], 'k-')
plt.xlim(1.0,8.0)
#plt.show()
'''
## single struct diff
'''
t0 = 566
f0 = traj[t0].get_positions()
F0 = rfft(np.asarray(f0[:,1]), norm='ortho')

t1 = 900
f1 = traj[t1].get_positions()
F1 = rfft(np.asarray(f1[:,1]), norm='ortho')

#print(F0[1], F1[1])
print(t1)
tot0 = np.sum(abs(F0))
tot1 = np.sum(abs(F1))
print('tot', tot0, tot1)
diff = 0.
dsq = 0.
for a in range(len(F0)):
    part0 = abs(F0[a])/tot0
    part1 = abs(F1[a])/tot1
#    print(part0*1e-2)
    print(part0, part1)
    if (part0 >= 1e-2):
        diff += abs(part0-part1)
        dsq += (part0-part1)**2
dsq = np.sqrt(dsq)
print(diff, dsq)
'''

# function reconstruction
'''
rec = []
for a in range(N):
    sum = 0.
    for k in range(N):
        sum+=F[k]/np.sqrt(N)*np.exp(2*np.pi*1j*k*a/N)
    rec.append(sum)

'''

#### average modes, mbd vs ts
'''
#Ambd, ERRmbd = get_avmodes(traj, t0)
# TS trajectory
print('done 1')
traj = TrajectoryReader('./TS_grad50-70/chain.traj')
Ats, ERRts = get_avmodes(traj, t0)
#Ftraj = do_Fourier(traj)

fig, axs = plt.subplots(2)
Nat = 40

w = [ 2*np.pi*k/Nat for k in range(1,int(Nat/2)) ]
msize = 5
axs[0].set_title('Average Fourier coeffs')
axs[0].errorbar(w, Ambd[1:int(Nat/2)], yerr = ERRmbd[1:int(Nat/2)], fmt = 'b.', markersize = msize)
axs[0].errorbar(w, Ats[1:int(Nat/2)], yerr = ERRts[1:int(Nat/2)], fmt = 'r.', markersize = msize)
axs[0].set_ylabel('A($\omega)$')
axs[1].plot(w, Ats[1:int(Nat/2)]-Ambd[1:int(Nat/2)], 'bo')
axs[1].text(1.0, 0.015, '$A_{TS}- A_{MBD}$')
axs[1].set_xlabel('$\omega$')
#plt.savefig('avA-grad70.pdf')
#print(Ambd)
#print(Ats)
plt.show()
'''




#if (os.path.isfile(name + '.xyz') == False):
#    traj2xyz(name)
#u = mda.Universe(name + '.xyz')
#print(len(u.trajectory))
