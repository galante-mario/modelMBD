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


