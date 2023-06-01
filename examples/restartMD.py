import numpy as np
from os import system
from ase.io import Trajectory

# setting a ring of 32 atoms

from structures import set_chain
Nat = 40    # number of atoms
n = 5
f = 2

pos = set_chain(Nat, n, f, 2)

model = 'MBD'   # vdW model

# dynamics setup

from ase_interface.dynamics import set_dynamics

n_steps = 4e4   # *2 fs = runtime

solver = set_dynamics(pos, 'MBD', 'nvt', T = 5.,    # T in Kelvin
                bathsId = 'all', restart= True)
# running dynamics
solver.run(n_steps)

told = Trajectory('chain.traj', 'a')
tnew = Trajectory('rst-chain.traj')
for atoms in tnew: told.write(atoms)
told.close()
system('sed -i 1d rst-chain.log')
logold = np.loadtxt('chain.log')
logrst = np.loadtxt('rst-chain.log')
logrst[:,0]+=logold[-1,0]
log = np.concatenate((logold, logrst), axis=0)
np.savetxt('chain.log', log)
system('rm rst-chain.traj rst-chain.log')


#--EOF--#
