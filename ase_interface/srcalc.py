# Calculator for the short range part of the model Hamiltonian.
# Relies on the class Hsr

import os
import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import all_changes
from short_range import Hsr

class sr_potential(Calculator):

    default_parameters = {
                'mode':'chain',
                'restrain_axis':[],       # restrain atoms along axis by k_r/2*<axis>^2
                'restrain_level':2.,      # restrain atoms with k_r = 2
                'reference_geom':[],      # geometry for axis restrained
                'reference_confs':[],     # mtd reference conformations 
                'restrained_atoms':[],    # restrain system bounds (chain only for now)
                'bounds_geom':[]          # atoms for system bounds
                         }

    implemented_properties = ['energy', 'forces']

    valid_args = ['mode',\
            'restrain_axis', 'restrain_level', 'restrained_atoms',\
            'reference_confs', 'reference_geom', 'bounds_geom']

    valid_modes = ['chain', 'ring', 'cluster', 'graphene']

    def __init__(self, restart=None, atoms=None, ignore_bad_restart_file=False,\
                label=os.curdir, **kwargs):

        for arg, val in self.default_parameters.items(): setattr(self, arg, val)

        for arg, val in kwargs.items():

            if arg == 'mode' and val not in self.valid_modes:
                errtxt  = "Available values of 'mode' are '"
                errtxt += "', '".join(self.valid_modes)+"'"
                raise ValueError(errtxt)

            if arg in self.valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown keyword arg "%s": not in %s'
                                   %(arg, self.valid_args))

        restraint_OK1 = all([axis in ['x','y','z'] for axis in self.restrain_axis])
        restraint_OK2 = all([axis in [0, 1, 2] for axis in self.restrain_axis])
        if restraint_OK1:
            self.restrain_axis = [self.ax2dim[ax] for ax in self.restrain_axis]
        elif restraint_OK2:
            pass
        else:
            errtxt  = "Elements of 'restrain_axis' have to be from "
            errtxt += "['x', 'y', 'z'] or [0, 1, 2]"
            raise RuntimeError(errtxt)

        self.with_restraint = (len(self.restrain_axis) > 0)

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)


    def get_potential_energy(self, atoms=None):
        self.calculate(atoms)
        return self.results['energy']

    def get_forces(self, atoms=None):
        self.calculate(atoms)
        return self.results['forces']

    def update_properties(self, atoms):
        if not hasattr(self, 'atoms') or self.atoms != atoms:
            self.nAtoms = len(atoms)
            self.symbols = atoms.get_chemical_symbols()
#            self.build_interaction_lists()
#            for par in self.tensor_args:
#                self.check_interaction_params(atoms, to_check=par)

    def calculate(self, atoms, properties=['energy', 'forces'], system_changes=all_changes):

        self.update_properties(atoms)

        if self.mode == 'cluster':
            H0 = Hsr(mode='none')
        elif type(self.mode) == str:
            H0 = Hsr(mode=self.mode)
        else:
            raise NotImplementedError('custom bonds not implemented yet')

        pos = atoms.positions

        E = H0.energy(pos)
        F = H0.force(pos)

        if self.with_restraint:
            for ax in self.restrain_axis:
                if ax == 2:
                    dpos = pos[:,ax]
                else:
                    dpos = pos[:,ax] - self.reference_geom[:,ax]
                krax = self.restrain_level * dpos
                krdax = np.dot(krax, dpos)
                F[:,ax] += -self.restrain_level * dpos
                E += krdax / 2.
        ## add bound restrain if requested

        idx = self.restrained_atoms
        if idx == None:  idx = []
        for a in range(len(idx)):
#                    print(a, pos[idx[a],:])
#                dpos = pos[idx[a]][:] - np.asarray(self.reference_geom[a][:])
            dpos = pos[idx[a]][:] - np.asarray(self.bounds_geom[a][:])
            norm = np.dot(dpos,dpos)
            F[idx[a],:] += -self.restrain_level*dpos
            E += self.restrain_level*norm/2.


        self.results['energy'] = E
        self.results['forces'] = F


    # check interaction params ???







#--EOF--#
