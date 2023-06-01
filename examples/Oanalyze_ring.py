from ase.io.trajectory import TrajectoryReader
from os import system
import numpy as np
import os.path
from scipy.linalg import eig as diag
from numpy.linalg import norm as norm
from scipy.fft import fft, rfft
import matplotlib.pyplot as plt
from setup_systems import run_mbd, get_hessian
from setup_systems import set_chain

from scipy.signal import argrelextrema

#import MDAnalysis as mda
#from MDAnalysis.coordinates.XYZ import XYZReader as reader

## temp forces modules
from ase.atoms import Atoms
from ase.calculators.harmonic import default_avg_a_div_a0
from ase.calculators.libmbd import MBD as MBDcalc
##

def get_branches(pos):

    Nat = len(pos)

    Nat = len(pos)
    posU = pos[:Nat//2+1,:]
    posD = np.append([pos[0,:]],pos[Nat//2:,:], axis=0)

    ## order points???
    return posU, posD

def get_Avtime(traj, dir):

    u, d = get_branches(traj[0].get_positions())
    '''
    plt.scatter(u[:,0], u[:,1])
    plt.scatter(d[:,0], d[:,1])
    plt.show()
    '''

    Au0 = np.asarray(rfft(u[:,dir], norm='ortho'))
    Ad0 = np.asarray(rfft(d[:,dir], norm='ortho'))

    time = []
    Au = []
    Ad = []

    dt=2e-3

    for t in range(1,len(traj)):

        u, d = get_branches(traj[0].get_positions()) 

        time.append(t*dt)

        At = np.asarray(rfft(u[:,dir], norm='ortho'))
        Au.append(abs(At)/abs(Au0))

        At = np.asarray(rfft(d[:,dir], norm='ortho'))
        Ad.append(abs(At)/abs(Ad0))
    
    Au = np.asarray(Au)
    Ad = np.asarray(Ad)

    return Au, Ad 

traj = TrajectoryReader('./ring.traj')
Au, Ad = get_Avtime(traj, 1)
#time = np.linspace(0, 2e-3
fig, axs = plt.subplots(2,4)
axs[0,0].plot(time, Au[:,1])
axs[0,1].plot(time, Ad[:,1])
plt.show()
