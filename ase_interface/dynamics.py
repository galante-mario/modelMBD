import numpy as np
from ase.units import fs, kB, Hartree, Bohr
from ase.atoms import Atoms
from vdw import set_parameters
from ase_interface.qmme import qmme
from ase.md import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io import read, Trajectory

# temporary calcH0 
def setup_calcH0(harmtype, restrId, bounds_geom, restrplane):

    axis = []

    
    from ase_interface.srcalc import sr_potential
    calcH0 = sr_potential(mode = harmtype, 
                          restrain_axis       = axis,
                          reference_geom      = [],
                          restrain_level      = 5,
                          restrained_atoms    = restrId,
                          bounds_geom = bounds_geom
                          )
    '''
    from ase_interface.harmonic import harmonic_potential
    calcH0 = harmonic_potential(mode                = harmtype,
                                k                   = 39.57,
                                R0                  = 1.52,
                                with_repulsion      = True,
                                Arep                = 1878.38,
                                gamma               = 5.49,
                                restrained_atoms    = restrId,
                                bounds_geom         = bounds_geom,
                                restrain_axis       = axis,
                                reference_geom      = [],
                                restrain_level      = 5
                                )
    '''
    return calcH0


def set_dynamics(pos, srmode, vdWmodel, ensemble, vdWparams='default', restart=False, mtd=False,
        bathsId=None, T=None, restrId=None, restrplane=None, basename=None, beta=0.81):

    dt = 2.*fs
    tau = 4e-3  # gamma

    H0 = srmode
    if np.linalg.norm(pos[0]-pos[-1]) <= 2.:
        H0 = 'ring'
    else:
        H0 = 'chain'

    if basename == None: basename = H0

    if restart:
        atoms = read(basename+'.traj', index=-1)
        pos = atoms.get_positions()
        Nat = len(pos)
    else:
        Nat = len(pos)
        atoms = Atoms('C'+str(Nat), positions=pos)
    
    extrema = np.argsort([row[0] for row in pos])

    if isinstance(restrId, np.ndarray):
        restrId = [extrema[0], extrema[-1]]

    bounds_geom = pos[restrId]

    if restrplane == None:
        axis = []
    else:
        axis = [restrplane]

    # sr calc
    from ase_interface.srcalc import sr_potential
    calcH0 = sr_potential(mode = srmode, 
                          restrain_axis       = axis,
                          reference_geom      = [],
                          restrain_level      = 5,
                          restrained_atoms    = restrId,
                          bounds_geom = bounds_geom
                          )

#    calcH0 = setup_calcH0(H0, restrId, bounds_geom, restrplane)

    # vdW calc
    params = set_parameters(pos, vdWparams)
    
    # general vdw calculator calling the vdw class to be done

    from ase_interface.pymbdcalc import pyMBD
    calc_vdW = pyMBD(mode='fortran', model=vdWmodel, custom_beta=beta, custom_vdW=params)
    if vdWmodel == 'MBD':
        calc_vdW = pyMBD(mode='fortran', model='MBD', custom_beta=beta, custom_vdW=params)
    elif vdWmodel == 'TS':
        calc_vdW = pyMBD(mode='fortran', model='TS', custom_beta=beta, custom_vdW=params)
    elif vdWmodel == 'novdW':
        calc_vdW = pyMBD(mode='fortran', model='novdW', custom_beta=beta, custom_vdW=params)
    else:
        raise RuntimeError(vdWmodel, 'not implemented')

    calc_vdW.get_forces(atoms=atoms)

    calc = qmme(
            atoms          = atoms,
            nqm_regions    = 1,
            nmm_regions    = 1,
            qm_atoms       = [[(0,Nat)]],
            mm_atoms       = [(0,Nat)],
            mm_mode        = 'allatoms',
            qm_calculators = [calcH0],
            mm_calculators = [calc_vdW],
            qm_pbcs        = [(False,False,False)],
            mm_pbcs        = [(False,False,False)],
           )
                        
    atoms.set_calculator(calc)

    trajname = basename + '.traj'
    logname = basename + '.log'
    if restart: 
        trajname = 'rst-'+trajname
        logname = 'rst-'+logname

    if ensemble == 'nve':

        solver = VelocityVerlet(atoms, dt, trajectory=trajname,
                logfile = logname)

    elif ensemble == 'nvt':

        if bathsId == None:
            Nbaths = max(1, int(Nat/20))
            bathsId = np.array([[extrema[:Nbaths]], [extrema[-Nbaths:]]])
        elif bathsId == 'all':
            bathsId = np.asarray(range(0,Nat))
        
        gamma = np.zeros((Nat,1))

        if T == None:
            raise RuntimeError('temperature required for nvt')
        elif type(T) == float or type(T) == int:
            Ts = T
            T = np.ones((Nat, 1))*Ts*kB
        elif len(T) == 2:
            Ts, Th = T[0], T[1]
            T = np.ones((Nat, 1))*Ts*kB
            if hasattr(bathsId[0], "__len__"):
                T[bathsId[1]] = Th*kB
        else:
            raise RuntimeError('unknown temperature format')
 
        gamma[bathsId.flatten()] = tau        

        MaxwellBoltzmannDistribution(atoms, kB*Ts)
            
        solver = Langevin(atoms, dt, T, gamma,
                    trajectory=trajname, logfile=logname)# fixcm=True)

    else:
        raise RuntimeError('unknown ensemble')

    return solver


