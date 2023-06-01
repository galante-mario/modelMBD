import numpy as np

model = 'MBD'   # vdW model
restart = False

# setting a ring of 32 atoms
if not restart:
    from structures import set_ring
    Nat = 32    # number of atoms
    pos = set_ring(Nat)

# constrained atoms every pi/2
bounds = [int(i*Nat/4) for i in range(4)]

# dynamics setup

from ase_interface.dynamics import set_dynamics

n_steps = 1e3   # *2 fs = runtime

solver = set_dynamics(pos, 'ring', 'MBD', 'nvt', T = 10,    # T in Kelvin
                restrId = bounds)
# running dynamics
solver.run(n_steps)

#--EOF--#
