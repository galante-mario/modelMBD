import sys
import numpy as np
from matplotlib import pyplot as plt
from numpy.linalg import norm as norm

from scipy.optimize import broyden1 as solve
from constants import chain_Ainit

import subprocess
import os

#### structural info from pdb

def pdb2groups(filein):
    os.system('grep "ATOM" '+filein+' > tmp')
    os.system('grep "HETATM" '+filein+' >> tmp')
    with open('tmp', 'r') as f:
        lines = f.readlines()
    os.system('rm tmp')
    Nat = len(lines)
    atid = np.empty(Nat, dtype='object')
    restype = np.empty(Nat, dtype='object')
    resnum = np.zeros(Nat, dtype='int')
    coor = np.zeros((Nat,3))
    species = np.empty(Nat, dtype='object')
    for l in range(Nat):
        line = lines[l].strip().split()
        i = int(line[1])-1
        atid[i] = line[2]
        restype[i] = line[3]
        resnum[i] = float(line[5])
        coor[i] = np.array(line[6:9], dtype=float)
        species[i] = line[11]
    return atid, restype, resnum, coor, species

#### model systems setup

def set_ring(Nat):
    from constants import harmonic
    R0 = harmonic['R0']
    phi_grid, dphi = np.linspace(0., 2.*np.pi, Nat, endpoint = False, retstep = True)
    R = R0/np.sin(dphi/2.)/2.
    pos = []
    for phi in phi_grid: pos.append([R*np.cos(phi), R*np.sin(phi), 0.])

    return np.asarray(pos)

def set_sphere(Nat):
    from constants import harmonic
    R0 = harmonic['R0']
    dphi = 3*np.pi/Nat
    R = 0.5*R0/np.sqrt( 0.5*(1 - np.cos(dphi)*np.sin(dphi)) )
    pos = []

    a = 4*np.pi*R/Nat
    d = np.sqrt(a)
    Nt = int(np.pi/d)
    dt = np.pi/Nt
    dp = a/dt
    i=0
    for m in range(Nt):
        theta = np.pi*(m+0.5)/Nt
        Np = int(2*np.pi*np.sin(theta)/dp)
        for n in range(Np):
            phi = 2*np.pi*n/Np
            pos.append(np.array([R*np.cos(phi)*np.sin(theta), 
                               R*np.sin(phi)*np.sin(theta),
                               R*np.cos(theta)]))       

    return np.array(pos)


def _integral(Nat, A, n, L, dim):

    from scipy.integrate import quad as integrate
    from utils.pynverse import inversefunc

    a = n*np.pi/L

    def amp(x, A):
        damp = np.exp(-abs(x-L/2)**(2)/L)
        return A*damp**(dim-2)

    def f(x):
        fy = amp(x, A)*np.sin(a*x)
        fz = amp(x, A)*np.cos(a*x)
        return np.array([fy, fz*(dim-2)])

    def fp2(x):
        b = (-2/L*np.abs(x-L/2.))*(dim-2)
        fy = amp(x, A)*( a*np.cos(a*x) + b*np.sin(a*x) )
        fz = amp(x, A)*(-a*np.sin(a*x) + b*np.cos(a*x) )
        return fy**2 + fz**(2*(dim-2))

    def D(x0):
        out = integrate(lambda x: np.sqrt(1 + fp2(x)), 0, x0)
        return out[0]
 
    x0 = inversefunc(D)
    
    pos = np.zeros((Nat,3))
    gamma = D(L)/(Nat-1)
    for at in range(1,Nat):
        xi = x0(gamma*at)
        pos[at,:] = np.insert(f(xi), 0, xi, axis=0)

    return np.asarray(pos)

def set_chain(Nat, n, f, dim):

    from constants import harmonic
    R0 = harmonic['R0']
    from scipy.integrate import quad as integrate
    from utils.pynverse import inversefunc

    L = R0*(Nat-1)/f
    a = n*np.pi/L

    # calculated for 2D structures
    def func(A):
        pos = _integral(Nat, A, n, L, dim)
        av = np.sum([norm(pos[i]-pos[i+1]) for i in range(Nat-1)])
        return av/(Nat-1)-R0
    p = chain_Ainit[n]
    A0 = (Nat-1)*p[0] + L*p[1] + (Nat-1)*np.sqrt(R0-L/(Nat-1))*p[2]
    A = solve(func, A0)

    pos = _integral(Nat, A, n, L, dim)

    return pos

#### graphene setup

def set_graphene(Nat):
    
    from math import modf
    from constants import graphene

    f, i = modf(np.sqrt(Nat/2+1))
    nc = int(i)
    Nat = 2*(nc**2-1)
    if f!=0: print('warning: Nat changed to',Nat)

    a = graphene['R0']

    # lattice vectors
    v1 = 0.5*a*np.array([3, np.sqrt(3), 0])
    v2 = 0.5*a*np.array([3,-np.sqrt(3), 0])
    v = np.array([v1,v2])

    n, m = np.meshgrid(np.arange(nc),np.arange(nc))
    nm = np.array([n.flatten(), m.flatten()]).swapaxes(0,1)
    A = np.dot(nm,v)
    B = np.array([a,0,0])+A
    pos = np.ravel([A,B],'F').reshape(3,Nat+2).swapaxes(0,1)
    pos = pos[1:-1]
#    print(np.where(pos == pos +v1))

    return pos

def get_graphene_bonds(Nat):

    if isinstance(Nat, np.ndarray): Nat = len(Nat)
    pos = set_graphene(Nat)

    from constants import graphene
    from math import modf

    a = graphene['R0']
    v1 = 0.5*a*np.array([3, np.sqrt(3), 0])
    v2 = 0.5*a*np.array([3,-np.sqrt(3), 0])

    R = pos[:,None,:] - pos[None,:,:]
    # bond length
    L = np.linalg.norm(R, axis=2)
    # bond directions
    L[L==0] = 1e15  # takes care of division by 0
    r = R/L[:,:,None]

    nn = np.array(np.where(L==a)).swapaxes(0,1)
    bonds = np.unique(np.sort(nn,axis=1),axis=0)

    angs = np.zeros((Nat**2,3), dtype=int)
    diheds = np.zeros((Nat**2,2), dtype=int)
    a0 = np.zeros((Nat**2,2))
    Nang = 0 ; Ndih = 0
    nns = np.array([nn[nn[:,0]==i,1] for i in range(Nat)])

    for i in range(len(pos)):
        # looking for the right 2nn
        for v in [v1,v2]:
            # array correspondent to v1 shift is now 0,0,0
            d = pos - (pos[i] + v)
            k = np.where(~d.any(axis=1))[0]
            if len(k): 
                k = k[0]
                j = np.intersect1d(nns[i], nns[k])[0]
                angs[Nang] = np.array([i,j,k])
#                theta[Nang] = np.arccos(np.dot(r[i,j], r[j,k]))
                Xijk = np.cross(r[i,j], r[j,k])
                Tijk = np.arccos(np.dot(r[i,j], r[j,k]))
                ndih=0
                for l in nns[k]:
                    if l != j: 
                        diheds[Nang,ndih] = l
                        Xjkl = np.cross(r[j,k], r[k,l])
                        Tjkl = np.arccos(np.dot(r[j,k], r[k,l]))
                        ratio = np.dot(Xijk,Xjkl)/np.sin(Tijk)/np.sin(Tjkl)
                        a0[Nang,ndih] = np.arccos(ratio)
                        ndih+=1
                Nang+=1
    return bonds, angs[:Nang], diheds[:Nang], a0[:Nang]

def get_graphene_edges(Nat):

    if isinstance(Nat, np.ndarray): Nat = len(Nat)
    pos = set_graphene(Nat)

    from constants import graphene

    L = np.linalg.norm(pos[:,None,:]-pos[None,:,:], axis=2)

    nn = np.array(np.where(L==graphene['R0'])).swapaxes(0,1)
    nns = np.array([nn[nn[:,0]==i,1] for i in range(Nat)])
    ids = np.zeros(Nat, dtype=int)
    pos_edges = np.zeros((Nat,3))
    Nedges = 0
    for i in range(Nat):
        nni = nns[i]
        if len(nni) == 2: 
            ids[Nedges] = i
            Nedges+=1
    ids = ids[:Nedges]

    return ids, pos[ids] 

# coordinate system conversions
def cartesian2polar(pos):
    
    Nat = len(pos)

    polar = np.zeros((Nat-1,3))
    for i in range(Nat-1):
        r = pos[i+1,:]-pos[i,:]
        d = norm(r)
        theta = np.arctan2(r[0], r[1])
        phi = np.arcsin(r[2]/d)
        polar[i,:] = np.array([d, theta, phi])

    return polar

def polar2cartesian(polar):

    Nat = len(polar)+1

    pos = np.zeros((Nat,3))

    for i in range(Nat-1):
        d, theta, phi = polar[i]
        pos[i+1,:] = pos[i,:] + [
                     d*np.sin(theta)*np.cos(phi),
                     d*np.cos(theta)*np.cos(phi),
                     d*np.sin(phi)]

    return pos

def torsion_angles(pos):

    r = pos[1:]-pos[:-1]
    r = r/(norm(r,axis=1)[:,None])

    dots = np.multiply(r[1:], r[:-1]).sum(1)
    cross = np.cross(r[1:], r[:-1])

    bond = np.arccos(dots) 
    dihedral = np.multiply(cross[:-1],cross[1:]).sum(1)/(
                np.sin(bond[1:])*np.sin(bond[:-1]))

    bond[bond<=-1e-10]+=2.
    dihedral[dihedral<=-1e-10]+=2.
                
    return bond*np.pi, dihedral*np.pi

