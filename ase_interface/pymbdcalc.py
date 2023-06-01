import numpy as np
import os
import warnings
from pymbd import mbd_energy as MBDcalc_Py, from_volumes
from ase import atoms
from ase.units import Bohr, Hartree
from ase.calculators.calculator import Calculator


def _warnonlywarning(message, category=UserWarning, filename='', lineno=-1, file=None, line=None):
    print(message)

warnings.showwarning = _warnonlywarning


modes_avail = ['python']
try:
    from pymbd.fortran import MBDGeom as MBDcalc_F
    modes_avail.append('fortran')
except ImportError:
    warnings.warn("Failed to import FORTRAN module.")

try:
    from pymbd.tensorflow import MBDEvaluator as MBDcalc_TF
    modes_avail.append('tensorflow')
except ImportError:
    warnings.warn("Failed to import TensorFlow module.")


beta_parameters = {"other":0.81, "pbe":0.83,   "pbe0":0.85, "hse":0.85}
sR_parameters   = {"pbe":0.94,   "pbe0":0.96, "hse":0.96, 
                   "b3lyp":0.84, "am05":0.84, "blyp":0.62, 
                   "revpbe":0.60}

def beta_from_xc(xcf):
    try:
        return beta_parameters[xcf.lower()]
    except KeyError:
        warnings.warn("beta-parameter for "+xc+" functional not known. Using 1.0")
        return 1.0


def sR_from_xc(xcf):
    try:
        return sR_parameters[xcf.lower()]
    except KeyError:
        warnings.warn("s_R-parameter for "+xc+" functional not known. Using 1.0")
        return 1.0



default_parameters = {
                      'model':'MBD',
                      'xc':'other',
                      'Sr':0.94,
                      'n_omega_SCS':15,
                      'kgrid':(3,3,3),
                      'get_MBD_spectrum':False,
                      'mode':'fortran',
                      'calc_forces':True,
                      'calc_hessian':False,
                      'a_div_a0':0.87
                     }



class pyMBD(Calculator):
    """
    
    Many-Body Dispersion calculator class
    
    Interface to libmbd (https://github.com/jhrmnn/libmbd.git),
    featuring various implementations:
        . FORTRAN: MPI/OpenMP/(Sca)LAPACK support with access to
            energy, forces, stress, eigenfrequencies, eigenmodes
            with and without periodic boundary conditions
        . TensorFlow: access to forces, hessian (non-periodic)
        . Python: access to energy, (numerical) forces and stress
            with and without periodic boundary conditions
    
    by Martin Stoehr (martin.stoehr@uni.lu), Aug 2019.
    Modified by Mario Galante, Jan 2021 (custom parameters)
    
    """
    
    implemented_properties = ['energy', 'forces', 'stress', 'hessian']
    
    valid_args = ['model',\
                  'xc', \
                  'Sr', \
                  'n_omega_SCS', \
                  'kgrid', \
                  'get_MBD_spectrum', \
                  'mode', \
                  'calc_forces', \
                  'calc_hessian', \
                  'custom_beta', \
                  'custom_vdW'
                 ]
    
    
    def __init__(self, restart=None, ignore_bad_restart_file=False, \
                 label=os.curdir, atoms=None, **kwargs):
        
        ## set default arguments
        for arg, val in default_parameters.items():
            setattr(self, arg, val)
        
        ## set or overwrite any additional keyword arguments provided
        for arg, val in kwargs.items():
            if arg in self.valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown keyword arg "%s" : not in %s'
                                   % (arg, self.valid_args))
        
        if self.mode.lower() in ['f90', 'fortran']:
            self.mode = 'fortran'
        elif self.mode.lower() in ['tf', 'tensorflow']:
            self.mode = 'tensorflow'
        elif self.mode.lower() in ['py', 'python']:
            self.mode = 'python'
        else:
            msg = "'mode' has to be in ['fortran', 'tensorflow', 'python']."
            raise ValueError(msg)
        
        if self.mode not in modes_avail:
            msg =  self.mode.title()+" implementation not available "
            msg += "(for error log, see above)."
            raise ValueError(msg)
        
        Calculator.__init__(self, restart, ignore_bad_restart_file,
                            label, atoms, **kwargs)
        
    
    def get_potential_energy(self, atoms=None):
        """ Return dispersion energy as obtained by MBD calculation. """
        self.update_properties(atoms)
        return self.results['energy']
        
    
    def get_forces(self, atoms=None):
        if not self.calc_forces:
            raise ValueError("Please, specify 'calc_forces=True'.")
        if atoms==None:
            raise ValueError('atoms object not defined!')
        
        self.update_properties(atoms)
        return self.results['forces']
        
    
    def get_stress(self, atoms=None):
        if not self.calc_forces:
            raise ValueError("Please, specify 'calc_forces=True'.")
        
        self.update_properties(atoms)
        return self.results['stress']
        
    
    def get_hessian(self, atoms=None):
        if self.mode != 'tensorflow':
            msg = "Hessian only available in Tensorflow mode for now."
            raise NotImplementedError(msg)
        
        if not self.calc_hessian:
            raise ValueError("Please, specify 'calc_hessian=True'.")
        
        self.update_properties(atoms)
        return self.results['hessian']
        
    
    def update_properties(self, atoms):
        if not hasattr(self, 'atoms') or self.atoms != atoms:
            self.calculate(atoms)
        
    
    def calculate(self, atoms):
        self.atoms = atoms.copy()

        pos = np.asarray(atoms.get_positions())

        self._set_vdw_parameters(atoms=self.atoms)
        self._set_coordinates(atoms=self.atoms)
        if self.mode == 'fortran':
            self._run_MBD_f()
        elif self.mode == 'tensorflow':
            if self.periodic:
                msg =  "Periodic boundary conditions not supported by "
                msg += "TensorFlow implementation.\n"
                msg += "Please, use mode='fortran'."
                raise NotImplementedError(msg)
            else:
                self._run_MBD_tf()
        elif self.mode == 'python':
            self._run_MBD_py()
        
    
    def _set_vdw_parameters(self, atoms=None):
        
        if hasattr(self, 'custom_beta'):
            self.beta = self.custom_beta
        else:
            self.beta = beta_from_xc(self.xc)
       
        if hasattr(self, 'custom_vdW'):
            self.alpha0_TS, self.C6_TS, self.RvdW_TS = self.custom_vdW
            
        else:
            self.alpha0_TS, self.C6_TS, self.RvdW_TS = \
                from_volumes(atoms.get_chemical_symbols(), self.a_div_a0)
        
    
    def _set_coordinates(self, atoms=None):
        self.xyz = atoms.positions/Bohr
        if any(atoms.pbc):
            [a, b, c] = atoms.get_cell()
            V = abs( np.dot(np.cross(a, b), c) )
            if V < 1e-2: warnings.warn("Cell volume < 0.01 A^3")
            self.UC = atoms.get_cell()/Bohr
            self.periodic = True
        else:
            self.UC, self.kgrid, self.periodic = None, None, False
        
    def _get_py_forces(self):

        h = 1.e-3

        F = np.zeros((len(self.xyz),3))
        #print(self.xyz)
        for i in range(len(self.xyz)):
            for x in range(3):
                Ep = 0. ; Em = 0.
                posp = np.copy(self.xyz)
                posm = np.copy(self.xyz)
                posp[i,x] += h
                posm[i,x] -= h
                MBDp = MBDcalc_F(posp, lattice=self.UC, k_grid=self.kgrid,
                     n_freq=self.n_omega_SCS, get_spectrum=False)
                MBDm = MBDcalc_F(posm, lattice=self.UC, k_grid=self.kgrid,
                     n_freq=self.n_omega_SCS, get_spectrum=False)
#                print(MBDp.coords)
#                print(MBDm.coords)
                if self.model == 'MBD':
                    Ep = MBDp.mbd_energy(self.alpha0_TS, self.C6_TS, self.RvdW_TS, 
                        self.beta, force=False)
                    Em = MBDm.mbd_energy(self.alpha0_TS, self.C6_TS, self.RvdW_TS, 
                        self.beta, force=False)
                elif self.model == 'TS':
                    Ep = MBDp.ts_energy(self.alpha0_TS, self.C6_TS, self.RvdW_TS, 
                                    self.Sr)
                    Em = MBDm.ts_energy(self.alpha0_TS, self.C6_TS, self.RvdW_TS, 
                                    self.Sr)
                elif self.model == 'novdW': 
                    Ep = 0.; Em = 0.
#                print(i,x, Ep, Em)
                F[i][x] = (Ep - Em)/h/2.
#                F[i][x] = -(Ep - Em)/h/Bohr*Hartree

        return F

    def _get_TS_forces(self):

        N = len(self.xyz)
        F = np.zeros((len(self.xyz),3))
        d = 20.
        for i in range(N):
            for j in range(i+1,N):
                Rij = self.xyz[i] - self.xyz[j]
                r = np.linalg.norm(Rij)
                beta = self.Sr*(self.RvdW_TS[i]+self.RvdW_TS[j])
                edamp = np.exp(-d*(r/beta -1.))
                f = 1./(1.+edamp)
                Eij = -f*self.C6_TS[i]/r**6
                Fij = (d/beta*f*edamp -6./r)*Eij*Rij/r
                F[i,:] += Fij
                F[j,:] -= Fij
        return F
    
    def _run_MBD_f(self):
        """
        Run MBD calculation via FORTRAN implementation.
        """
        MBD = MBDcalc_F(self.xyz, lattice=self.UC, k_grid=self.kgrid,
                 n_freq=self.n_omega_SCS, get_spectrum=self.get_MBD_spectrum)

        if self.model == 'MBD':
            res = MBD.mbd_energy(self.alpha0_TS, self.C6_TS, self.RvdW_TS, 
                    self.beta, force=self.calc_forces)
        elif self.model == 'TS':
            res = [MBD.ts_energy(self.alpha0_TS, self.C6_TS, self.RvdW_TS,
                    self.Sr)]
            if self.calc_forces: 
                res.append(self._get_TS_forces())
#                res.append(self._get_py_forces())
        elif self.model == 'novdW':
            res = [0.]
            if self.calc_forces: res.append(np.zeros((len(self.xyz),3)))

        if self.periodic and self.get_MBD_spectrum and self.calc_forces:   # all
            ((self.results['energy'], self.MBDevals, self.MBDmodes), \
                    self.results['forces'], self.results['stress']) = res
            self.results['forces'] *= -1.*Hartree/Bohr
            self.results['stress'] *= -1.*Hartree/Bohr
        elif self.periodic and self.get_MBD_spectrum:   # no forces
            (self.results['energy'], self.MBDevals, self.MBDmodes) = res
        elif self.periodic and self.calc_forces:   # no spectrum
            (self.results['energy'], self.results['forces'], \
                    self.results['stress']) = res
            self.results['forces'] *= -1.*Hartree/Bohr
            self.results['stress'] *= -1.*Hartree/Bohr
        elif self.get_MBD_spectrum and self.calc_forces:   # no PBC
            ((self.results['energy'], self.MBDevals, self.MBDmodes), \
                    self.results['forces']) = res
            self.results['forces'] *= -1.*Hartree/Bohr
        elif self.get_MBD_spectrum:   # no forces and no PBC
            (self.results['energy'], self.MBDevals, self.MBDmodes) = res
        elif self.calc_forces:   # no spectrum and no PBC
            (self.results['energy'], self.results['forces']) = res
            self.results['forces'] *= -1.*Hartree/Bohr
        else:   # only energy (with and without PBC)
            self.results['energy'] = res
        self.results['energy'] *= Hartree

        

    def _run_MBD_tf(self):
        """
        Run MBD calculation via TensorFlow implementation.
        """
        
        MBD = MBDcalc_TF(gradients=self.calc_forces, 
                         hessian=self.calc_hessian, 
                         nfreq=self.n_omega_SCS)
        
        res = MBD(coords=self.xyz, alpha_0=self.alpha0_TS, C6=self.C6_TS, 
                  R_vdw=self.RvdW_TS, beta=self.beta)
        
        if self.calc_forces and self.calc_hessian:
            self.results['energy'], self.results['forces'], \
                    self.results['hessian'] = res
            self.results['forces'] *= -1.*Hartree/Bohr
            self.results['hessian'] *= -1.*Hartree/Bohr/Bohr
        elif self.calc_forces:
            self.results['energy'], self.results['forces'] = res
            self.results['forces'] *= -1.*Hartree/Bohr
        elif self.calc_hessian:
            self.results['energy'], self.results['hessian'] = res
            self.results['hessian'] *= -1.*Hartree/Bohr/Bohr
        else:
            self.results['energy'] = res
        
        self.results['energy'] *= Hartree
        
    
    def _run_MBD_py(self):
        """
        Run MBD calculation via pure Python implementation.
        """
        
        self.results['energy'] = MBDcalc_Py(self.xyz, self.alpha0_TS, self.C6_TS, 
                                self.RvdW_TS, self.beta, lattice=self.UC, 
                                k_grid=self.kgrid, nfreq=self.n_omega_SCS)
        self.results['energy'] *= Hartree

        self.results['forces'] = self._get_py_forces()
        self.results['forces'] *=-1.*Hartree/Bohr
        
        #TODO: numerical forces ?and stress?
        
    
    def get_MBD_frequencies(self, atoms=None):
        """
        Returns the spectrum of MBD (eigen)frequencies in a.u.
        """
        if self.mode != 'fortran':
            msg = "MBD frequencies are only available in 'fortran' mode for now."
            raise NotImplementedError(msg)
        elif self.model == 'TS':
            raise NotImplementedError('eigenfrequencies not available for TS')
        elif not self.get_MBD_spectrum:
            msg = "Please, specify 'get_MBD_spectrum=True' when initializing the calculator."
            raise ValueError(msg)
        elif not hasattr(self, 'MBDevals'):
            if not hasattr(self, 'atoms') and atoms is None:
                msg = "Please, specify atoms on input or run get_potential_energy() first"
                raise ValueError(msg)
            
            self.update_properties(atoms)
            
        return self.MBDevals
        
    
    def get_MBD_modes(self):
        """
        Returns the MBD (eigen)modes
        """
        if self.mode != 'fortran':
            msg = "MBD modes are only available in 'fortran' mode for now."
            raise NotImplementedError(msg)
        elif self.model == 'TS':
            raise NotImplementedError('eigenmodes not available for TS')
        elif not self.get_MBD_spectrum:
            msg = "Please, specify 'get_MBD_spectrum=True' when initializing the calculator."
            raise ValueError(msg)
        elif not hasattr(self, 'MBDmodes'):
            if not hasattr(self, 'atoms') and atoms is None:
                msg = "Please, specify atoms on input or run get_potential_energy() first"
                raise ValueError(msg)
            
            self.update_properties(atoms)
            
        return self.MBDmodes
        
    

#--EOF--#
