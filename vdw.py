import sys
import numpy as np
from numpy.linalg import norm as norm
from ase.units import fs, kB, Hartree, Bohr
from scipy.special import erf as erf
from itertools import product

def set_parameters(pos, parameters='default'):

    '''
    returns vdW TS parameters inclusive of volumes rescaling
    given a set of positions
    '''

    Nat = len(pos)


    if isinstance(parameters, str):

        mode = parameters

        if mode == 'default':
            # all C atoms
            a0, C6, Rvdw, vols = 12.0, 46.6, 3.59, 0.87
        else:
            raise RuntimeError('unknown preset parameter mode ', mode)

        a0l   = a0*np.ones(Nat)
        C6l   = C6*np.ones(Nat)
        Rvdwl = Rvdw*np.ones(Nat)
        V     = vols*np.ones(Nat)

    elif len(parameters) == 4:
    #elif isinstance(parameters, list) and len(parameters) == 4:
        par = parameters[0]
        if hasattr(par, '__len__') and len(par)==Nat:
            a0l, C6l, Rvdwl, V = np.asarray(parameters)
        elif isinstance(par,float):
            a0, C6, Rvdw, vols = parameters
            a0l   = a0*np.ones(Nat)
            C6l   = C6*np.ones(Nat)
            Rvdwl = Rvdw*np.ones(Nat)
            V     = vols*np.ones(Nat)
        else:
            raise RuntimeError('uncorrect list format')
    else:
        raise RuntimeError('unknown parameters format')

    return a0l*V, C6l*V**2, Rvdwl*V**(1/3)


default = {
            'beta':0.81,             # MBD damping factor
            'Sr':0.94,               # TS  damping factor
            'screening':'plain',     # MBD screening mode
            'parameters':'default',  # vdW parameters
          }

class vdWclass:

    valid_args = ['beta',\
                  'Sr',\
                  'screening',\
                  'parameters',\
                 ]

    valid_models = ['TS', 'MBD', 'TSerf', 'MBDerf', 'novdW']

    from pymbd.fortran import MBDGeom as evaluator

    def __init__(self, **kwargs):

        for arg, val in default.items():
                setattr(self, arg, val)

        for arg, val in kwargs.items():
            if arg in self.valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown value "%s" for argument %s' 
                        % (arg, self.valid_args))

        self.calc = self._set_calc()

    def _combineC6(self):
       
        r = self.a0[:,None]/self.a0[None,:]
        d = self.C6[:,None]/r + self.C6[None,:]*r
        return 2*self.C6[:,None]*self.C6[None,:]/d


    def _get_energy(self, calc):

        if self.model == 'MBD':
            out = calc.mbd_energy(self.a0, self.C6, self.Rvdw, self.beta, 
                                force = False, variant = self.screening)
        elif self.model == 'MBDerf':
            out = calc.mbd_energy(self.a0, self.C6, self.Rvdw, self.beta, 
                        damping='dip,gg',force = False, variant = self.screening)
        elif self.model == 'TS':
            d = 20.
            C6ij = self._combineC6()
            beta = self.Sr*(self.Rvdw[:,None]+self.Rvdw[None,:])
            Rij = self.pos[:,None,:]-self.pos[None,:,:]
            rij = np.linalg.norm(Rij, axis=2)
            f = 1./(1. + np.exp(-d*(rij/beta-1.)))
            rij[rij==0]=1e20
            Eij = -f/rij**(6)*C6ij
            out = np.sum(np.triu(Eij,1).flatten())
        elif self.model == 'TSerf':
            sigma = (np.sqrt(2/np.pi)*self.a0/3)**(1/3)
            sij = np.sqrt(sigma[:,None]**2 + sigma[None,:]**2)
            Rij = (self.pos[:,None,:]-self.pos[None,:,:])/Bohr
            rij = np.linalg.norm(Rij, axis=2)
            eta = rij/sij
            rij[rij==0] = 1e20
            Eij = -4/np.sqrt(np.pi)/sij**3*np.exp(-eta**2) - 2/rij**3*(
                    erf(eta)-2/np.sqrt(np.pi)*eta*np.exp(-eta**2))
            out = np.sum(np.triu(Eij,1).flatten())/Hartree
        elif self.model == 'novdW':
            out = 0.

        else:
            raise RuntimeError('unknown model')

        if isinstance(out, float):
            return out*Hartree
        else:
            return out[0]*Hartree, out[1], out[2]

    def _get_forces(self, calc):

        if self.model == 'MBD':
            E, F = calc.mbd_energy(self.a0, self.C6, self.Rvdw, self.beta,
                                force = True, variant = self.screening)

        elif self.model == 'TS':
            d = 20.
            C6ij = self._combineC6()
            beta = self.Sr*(self.Rvdw[:,None]+self.Rvdw[None,:])
            Rij = self.pos[:,None,:]-self.pos[None,:,:]
            rij = np.linalg.norm(Rij, axis=2)
            f = 1./(1. + np.exp(-d*(rij/beta-1.)))
            temp = np.copy(rij)
            temp[temp==0] = 1e20
            invrij = temp**(-1)
            Eij = - f*C6ij*invrij**(6)
            bracket = d/beta*f*np.exp(-d*(rij/beta-1.)) - 6.*invrij
            Fij = bracket[:,:,None]*Eij[:,:,None]*invrij[:,:,None]*Rij
            if forceij:
                F = Fij
            else:
                F = np.sum(Fij,axis=1)
        elif self.model == 'TSerf':
            sigma = (np.sqrt(2/np.pi)*self.a0/3)**(1/3)
            sij = np.sqrt(sigma[:,None]**2 + sigma[None,:]**2)
            Rij = (self.pos[:,None,:]-self.pos[None,:,:])/Bohr
            rij = np.linalg.norm(Rij, axis=2)
            eta = rij/sij
            rij[rij==0] = 1e20
            Fij = 2/rij**4*( erf(eta)-2*eta/np.sqrt(np.pi)*np.exp(-eta**2) ) -8/3/np.sqrt(np.pi)/rij/sij**5*np.exp(-eta**2)*(rij**2+sij**2)
            Fij = Fij[:,:,None]*Rij/rij[:,:,None]
            F = np.sum(Fij, axis=1)
        elif self.model == 'novdW':
            F = np.zeros((len(self.pos),3))
        return -F*Hartree/Bohr

    def _get_T(self, R, S):
        rij = norm(R, axis=2)
        damp = 1./(1.+np.exp(-6.*(rij/S-1.)))
        num1 = -3*R[:,:,:,None]*R[:,:,None,:]
        num2 = rij[:,:,None,None]**2*np.eye(3)[None,None,:,:]
        r5 = rij[:,:,None,None]**5
        r5[r5==0.] = 1.e-10
        return damp[:,:,None,None]*(num1+num2)/r5
        
    def _get_relay_matrix(self, calc, shape, form):

        Nat = len(self.pos)

        S = self.beta*(self.Rvdw[:,None]+self.Rvdw[None,:])

        diag = np.concatenate([ np.ones(3)*a0**(-1) for a0 in self.a0 ])

        dists = self.pos[:,None,:] - self.pos[None,:,:]
        rij = norm(dists, axis=2)
        damp = 1./(1.+np.exp(-6.*(rij/S-1.)))

        # -3*r(i,j,a)*r(i,j,b)
        num1 = -3*dists[:,:,:,None]*dists[:,:,None,:]
        # r(i,j)**2 \delta(a,b)
        num2 = rij[:,:,None,None]**2*np.eye(3)[None,None,:,:]
        r5 = rij[:,:,None,None]**5
        r5[r5==0.] = 1.e-10
        # damp*(num1+num2)/r**5
        odiag = damp[:,:,None,None]*(num1[:,:,:,:]+num2[:,:,:,:])/r5

        invB = np.empty((0,3*Nat),float)
        for a in range(3):
            for i in range(Nat):
                row = np.concatenate([ odiag[i,:,a,b] for b in range(3)])
                invB = np.append(invB, [row],axis=0)
        np.fill_diagonal(invB,diag)
        np.savetxt('old_a.txt', invB)
        B = np.linalg.inv(invB)

        if shape == 'tensor':
            tensor = np.zeros((3,3,Nat,Nat))
            for a in range(3):
                for b in range(3):
                    tensor[a,b,:,:] = B[a*Nat:(a+1)*Nat, b*Nat:(b+1)*Nat]
            B = tensor
        elif shape != 'matrix':
            raise NotImplementedError('unknown polmat_shape')
        return B

    def _get_hessian(self):

        Nat = len(self.pos)
        hess = np.zeros((3*Nat, 3*Nat))
        h = 1e-3
        F = np.zeros((3*Nat, 2, 3*Nat))
        for i in range(Nat):
            for j in range(3):
                for s in range(2):
                    pos = self.pos
                    pos[i,j]+=(-1)**s*h
                    calc = self.calc(pos)
                    f = self._get_forces(calc)
                    F[i*3+j,s] = -f.flatten()
        hess[:] = (F[:,0] - F[:,1])*0.5/h

        return hess

    def _optimise(self, dim, tol, maxiter):

        Nat = len(self.pos)

        from molecular_bonds import get_H0_chain
        # standard chain only!
        from short_range import Hsr
        hcov = Hsr()
        print(hcov.bond_type)
        bid = [0, Nat-1]
        bpos = self.pos[bid]
        posvar = self.pos[:,:dim]
        posvar = np.delete(posvar, bid, axis=0)

        par0 = set_parameters(self.pos, self.parameters)
        self.parameters = np.delete(par0, bid, axis=1)

        def func(var):
            posvarit = np.reshape(var, (Nat-len(bid),dim))
            self.pos = np.insert(posvarit, [0], bpos[0], axis=0)
            self.pos = np.append(self.pos, [bpos[1]], axis=0)
            # conversion to ang for the harmonic potential
            E0 = get_H0_chain(self.pos*Bohr)
            Ecov = hcov.energy(self.pos*Bohr) 
            print(E0, Ecov)
            calc = self.calc(self.pos)
            Evdw = self._get_energy(calc)
            return E0 + Evdw

        E = func(posvar.flatten())
        from scipy.optimize import minimize
        print('Starting energy', E)

        outs = minimize(func, posvar.flatten(), method = 'CG', 
                        tol=tol, options={'maxiter': maxiter, 
                        'disp':True,'return_all':True})

        print('Result:', outs.success, outs.nit)
        print('E_out:', outs.fun)

        self.parameters = par0

        steps = []
        with open('steps'+self.model+'.txt', 'w') as f:
            for i in range(outs.nit):
                posi = np.reshape(outs.allvecs[i], (Nat-len(bid),dim))
                posi = np.insert(posi, [0], bpos[0], axis=0)
                posi = np.append(posi, [bpos[1]], axis=0)
                np.savetxt(f, posi*Bohr)
                f.write('\n \n')
                steps.append(posi*Bohr)
        steps = np.array(steps)

        return steps
       


    def calculate(self, pos, model, what='energy',
                    polmat_shape='matrix',
                    opt_dim=3, opt_tol=1e-3, opt_maxiter=5000,
                    mbd_evecs=True):

        # coordinates from ang to Bohr
        self.pos = pos/Bohr
        # vdw model
        self.a0, self.C6, self.Rvdw = set_parameters(
                                        pos, self.parameters)
        if model not in self.valid_models:
            raise ValueError('model'+model + 'not known')
        self.model = model

        if isinstance(what, str):
            if model == 'MBD' and what == 'spectrum':
                calc = self.calc(self.pos, get_spectrum=True)
                out = self._get_energy(calc)
                # only outputs eigenvalues and eigenvectors
                return out[0], out[1], out[2]
            else:
                what = [what]

        calc = self.calc(self.pos)

        outs = []


        for task in what:
            if   task == 'energy':
                outs.append(self._get_energy(calc))
            elif task == 'forces':
                outs.append(self._get_forces(calc, forceij=forceij))
            elif task == 'hessian':
                outs.append(self._get_hessian())
            elif task == 'optimise':
                outs.append(self._optimise(dim=opt_dim, tol=opt_tol, maxiter=opt_maxiter))
            elif task == 'polmat':
                if self.model == 'TS':
                    raise NotImplementedError(
                        task,' non available for model, ', self.model)
                if lattice is None:
                    outs.append(self._get_relay_matrix(calc, 
                        shape=polmat_shape, form=polmat_form))
                else:
                    outs.append(self._get_relay_matrix_lattice(calc, 
                        shape=polmat_shape, form=polmat_form, 
                        lattice=lattice, kps=kps))
            elif task == 'spectrum':
                if self.model == 'TS':
                    raise RuntimeError(
                        task,' non available for model, ', self.model)
                else:
                    raise RuntimeError(
                        'spectrum available only as single task')
            else:
                raise RuntimeError(item, ' is not implemented')

        if len(what) == 1: outs=outs[0]

        return outs

