from ase.atoms import Atoms
from ase.units import fs, kB
from warnings import warn
from os import system
import numpy as np
from numpy.linalg import norm as norm

from setup_systems import get_chain_fixL, get_chain_spaced

from matplotlib import pyplot as plt

Nat = 40
R0 = 1.52
nodes = 5
L0 = R0*(Nat-1)
f = 2.0

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fgs = 'gs3D40_n5_f4.0tol3.txt'
fts = 'TS3D40_n5_f4.0tol2.txt'
fmbd = 'MBD3D40_n5_f4.0tol3.txt'
lab = ['input', 'TS', 'MBD']
c = 0
for files in [fgs, fts, fmbd]:
    pos = []
    with open( files, 'r') as f:
        for i in range(Nat):
            line = f.readline()
            ns = line.strip()
            posline = ns.split()
            at = [float(j) for j in posline]
            pos.append(at)
    pos = np.asarray(pos)
    ax.scatter(pos[:,0], pos[:,1], pos[:,2], label = lab[c])
    ax.plot(pos[:,0], pos[:,1], pos[:,2])
    c+=1

X = np.arange(-0.5,15,0.01)
Y = np.arange(-6,6,0.01)
X, Y = np.meshgrid(X, Y)
Z = 0*(X+Y)
ax.plot_surface(X, Y, Z*0., color = '#2222dd', alpha = 0.1)

#ax.view_init(30,-75)
ax.view_init(13,-82)


ax.set_xlabel('x [ang]')
ax.set_ylabel('y [ang]')
ax.set_zlabel('z [ang]')
ax.legend()
plt.savefig('fig_minim2Da.pdf')
plt.show()

