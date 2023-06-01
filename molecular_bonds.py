import sys
import numpy as np
from numpy.linalg import norm as norm


def get_Eharmonic(pos):

    N = len(pos)

    k = 39.57
    R0 = 1.52
    shift = -1035.2

    Eh = 0.
    for i in range(N-1):
        Rij = norm(pos[i,:]- pos[i+1,:]) 
        Eh += k*(Rij-R0)**2/2. + shift
    return Eh

def get_Etot(pos, model, method='pymbd-f'):

    N = len(pos)

    Arep = 1878.38
    gamma = 5.49

    Eh = get_Eharmonic(pos)
    Erep = 0.
    for i in range(N-1):
        for j in range(i+2, N):
            Rij = norm(pos[i,:]- pos[j,:]) 
            Erep += Arep*np.exp(-gamma*Rij)
    
    from vdw import vdWclass
    vdwcalc = vdWclass()
    Evdw = vdwcalc.calculate(pos, model)

    return Eh + Erep + Evdw

def get_H0_chain(pos):

    N = len(pos)

    Arep = 1878.38
    gamma = 5.49

    Eh = get_Eharmonic(pos)

    Erep = 0.
    for i in range(N-1):
        for j in range(i+2, N):
            Rij = norm(pos[i,:]- pos[j,:]) 
            Erep += Arep*np.exp(-gamma*Rij)

    return Eh + Erep

def get_covalent_force(pos):
    
    N = len(pos)
    R0 = 1.52
    k = 39.57
    Arep = 1878.38
    gamma = 5.49

    F = np.zeros((N,3))

    for i in range(N-1):
        R = pos[i]-pos[i+1]
        F[i] += k*(norm(R)-R0)*R/norm(R)
        F[i+1] = -F[i]
    for i in range(N):
        for j in range(N):
            if abs(j)>=2:
                R = pos[i]-pos[j]
                F[i]-=Arep*gamma*np.exp(-gamma*norm(R))*R/norm(R)
    return F

