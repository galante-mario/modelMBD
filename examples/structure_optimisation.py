import numpy as np
# fixed inter-atomic distance
from constants import R0

# chain definition

Nat = 10    # atom number
n = 3       # corrugation number
f = 2.0     # compression factor
dim = 2     # dimensionality of the structure

from structures import set_chain

pos = set_chain(Nat, n, f, dim)

#model = 'MBD'
model = 'TS'

# initialisation of vdW class with 'default' parameters

from vdw import vdWclass

vdw = vdWclass(parameters='default')

# run optimisation starting with coordinate pos

opt_steps = vdw.calculate(pos, model, 
                        what='optimise', opt_maxiter=100)

np.savetxt('opt' + model + '.txt', opt_steps[-1])


