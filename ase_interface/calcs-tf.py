import numpy as np
import os
import warnings
from pymbd import mbd_energy as MBDcalc_Py, from_volumes
from ase import atoms
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator
from ase.calculators.harmonic import default_avg_a_div_a0

### calculators for vdW-MBD and TS MD simulations from tensorflow
# Mario Galante, August 2020

class vdwTF(Calculator):

    implemented_properties = ['energy', 'forces']
    
    valid_args = ['xc', \
                  'model', \
                  'calc_forces', \
                 ]
 
    def __init__(self, restart=None, ignore_bad_restart_file=False, \
                 label=os.curdir, atoms=None, **kwargs):
        for arg, val in default_parameters.items():
            setattr(self, arg, val)

        print('tf')
        
        ## set or overwrite any additional keyword arguments provided
        for arg, val in kwargs.items():
            if arg in self.valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown keyword arg "%s" : not in %s'
                                   % (arg, self.valid_args))

        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)


    def get_potential_energy(self, atoms=None):
        """ Return dispersion energy """
        self.update_properties(atoms)
        return self.results['energy']
        
    
    def get_forces(self, atoms=None):
        if not self.calc_forces:
            raise ValueError("Please, specify 'calc_forces=True'.")
        
        self.update_properties(atoms)
        return self.results['forces']
    
    def update_properties(self, atoms):
        if not hasattr(self, 'atoms') or self.atoms != atoms:
            self.calculate(atoms)
 
    def calculate(self, atoms):

        self.atoms = atoms.copy()
        self._set_coordinates(atoms=self.atoms)
        self._set_params()

        self._run_tf()


    def _set_params(self): 

        nAtoms = len(self.xyz)
        hvr = np.ones(nAtoms)*([default_avg_a_div_a0['C']]*nAtoms)
        volumes = np.array(hvr)
        self.beta = 0.83
        self.Sr = 0.94
        a0 = 12
        C6 = 46.6
        Rvdw = 3.59

        self.alpha0_TS = a0*volumes
        self.C6_TS = C6*volumes**2
        self.RvdW_TS = Rvdw*volumes**(1/3)

    def _set_coordinates(self, atoms=None):
        self.xyz = atoms.positions/Bohr
        self.UC, self.kgrid, self.periodic = None, None, False

    def _run_tf(self):

        if self.model == 'MBD':

            from mbd_tf import MBDEvaluator as Evaluator

        elif self.model == 'TS':

            sys.exit('non implemented yet')

        calc = Evaluator(hessians=False)
        self._set_params()
        E, F = calc(self.xyz, self.alpha0_TS, self.C6_TS, self.RvdW_TS, 0.87)

        self.results['energy'] = E*Hartree
        self.results['forces'] = -F*Hartree/Bohr








