# class definining the Hamiltonian for short range interactions

import sys
import numpy as np
from numpy.linalg import norm as norm


default = {
            'mode':'chain',         # type of short range interactions
            'reptype':'expharm',    # exp repulsion for harmonic bonds
            'bond_list':'chain',    # chain bonds
            'mtd_refs':[],          # no mtd reference conformations
            'mtd_k':1e-3,           # standard mtd pushing strength 
            'mtd_alpha':0.1         # standard width of mtd bias potential
          }

class Hsr:

    valid_args = ['reptype',\
                  'mode',\
                  'bond_list',\
                  'mtd_refs',\
                  'mtd_k',\
                  'mtd_alpha',
                 ]

    vaild_modes = ['chain', 'ring', 'graphene', 'cluster']

    def __init__(self, **kwargs):

        for arg, val in default.items():
                setattr(self, arg, val)

        for arg, val in kwargs.items():
            if arg in self.valid_args:
                setattr(self, arg, val)
            else:
                raise RuntimeError('unknown value "%s" for argument %s' 
                        % (arg, self.valid_args))

    def _get_bonds(self, pos):

        Nat = len(pos)
        nobonds = np.concatenate(np.array([np.linspace((i,i+1), (i, Nat-1), Nat-i-1, dtype=int) for i in range(Nat)]))

        if self.mode == 'cluster':
            bonds = []
        elif self.mode == 'chain':
            bonds = np.array([[i, i+1] for i in range(Nat-1)])
        elif self.mode == 'ring':
            bonds = np.array([[i, i+1] for i in range(Nat-1)])
            bonds = np.append(bonds, [[Nat-1,0]], axis=0)
        elif self.mode == 'graphene':
            from structures import get_graphene_bonds
            self.reptype = 'none'
            bonds = get_graphene_bonds(len(pos))
        elif type(self.bond_list) == str:
            raise NotImplementedError('unknown present bond configuration ', self.bond_list)


        if self.mode == 'graphene':
            return bonds
        else:
            for bond in bonds:
                mask = (nobonds==bond).all(axis=1)
                nobonds = nobonds[~mask] 
            return bonds, nobonds


    def energy(self, pos):
        
        # bonding
        #rij = norm(pos[:,None,:] - pos[None,:,:], axis=2)
        if self.mode == 'cluster':
            Eb = 0
        elif self.mode == 'graphene':
            from constants import graphene
            a = graphene['R0']
            kb, ka, kd = graphene['kb'], graphene['kang'], graphene['kdih']
            bonds, angs, diheds, dihed0 = self._get_bonds(pos)
            R = pos[:,None,:] - pos[None,:,:]
            L = np.linalg.norm(R, axis=2)
            Eang, Edih = 0., 0.
            nang=0
            for i,j,k in angs[:3]:
                rij = R[i,j]/L[i,j]
                rjk = R[j,k]/L[j,k]
                Tijk = np.arccos(np.dot(rij, rjk))
                Ei = ka/2*(np.cos(Tijk)-0.5)**2
                Eang += Ei
                #Xijk = np.cross(rij, rjk)
                ndih=0
                # NO DIHEDRALS INCLUDED
                '''
                for l in diheds[nang]:
                    rkl = R[k,l]/L[k,l]
                    Xjkl = np.cross(rjk, rkl)
                    Tjkl = np.arccos(np.dot(rjk, rkl))
                    ratio = np.dot(Xijk,Xjkl)/np.sin(Tijk)/np.sin(Tjkl)
                    Edih += kd/2*(np.arccos(ratio)- dihed0[nang,ndih])**2
                    ndih+=1
                '''
                nang+=1
            Eb = Ebl + Eang + Edih

        else:
            bonds, nobonds = self._get_bonds(pos)
            from constants import harmonic
            Rij = pos[bonds[:,0],:] - pos[bonds[:,1],:]
            d = np.sum((norm(Rij, axis=1)-harmonic['R0'])**2)
            Eb = 0.5*harmonic['k']*d + harmonic['shift']*len(Rij)

        # repulsion

        if self.reptype == 'expharm':
            from constants import rep_harmonic
#            rij = pos[
#            allR = pos[:,None,:]-pos[None,:,:]
            Rij = pos[nobonds[:,0],:]-pos[nobonds[:,1],:]
            r = norm(Rij, axis=1)
            Erep = rep_harmonic['A']*np.sum(np.exp(-rep_harmonic['b']*r))
        elif self.reptype == 'none':
            Erep = 0.
        else:
            raise NotImplementedError('reptype ' + self.reptype + ' unknown')

        # mtd bias

        if self.mtd_refs:
            Ebias = 0.
        else:
            from utils.rmsd import rmsd_qcp as rmsd
            delta = np.array([rmsd(pos, ref) for ref in self.mtd_refs])         
            Ebias = self.mtd_k*np.sum( np.exp(-self.mtd_alpha*delta**2))
#            dref = np.array([self.pos.flatten() - ref.flatten for ref in self.mtd_refs])

        return Eb + Erep + Ebias

    def force(self, pos):

#        print(pos)
        # bonding
        Fb = np.zeros((len(pos),3))
        if self.mode == 'graphene':
            from constants import graphene
            a = graphene['R0']
            kb, ka, kd = graphene['kb'], graphene['kang'], graphene['kdih']
            bonds, angs, diheds, dihed0 = self._get_bonds(pos)
            R = pos[:,None,:] - pos[None,:,:]
            L = np.linalg.norm(R, axis=2)
            bl = L[bonds[:,0], bonds[:,1]]
            Rbonds = pos[bonds[:,0],:] - pos[bonds[:,1],:]
            fij = kb*(1 - a/L[bonds[:,0],bonds[:,1]])
            Fb[bonds[:,0]] -= fij[:,None]*Rbonds
            Fb[bonds[:,1]] += fij[:,None]*Rbonds
            # to be done in numpy
            for i,j,k in angs[:]:

                rij = R[i,j]/L[i,j]
                rjk = R[j,k]/L[j,k]
                dot = np.dot(rij, rjk)
#                Tijk = np.arccos(dot)
                fijk = -ka*(dot-0.5)/L[i,j]/L[j,k]
                Fb[i] += fijk*(R[i,j] - dot*rij)
                Fb[j] += fijk*(R[i,j]-R[j,k] + dot*rij/L[i,j] - dot*rjk/L[j,k])
                Fb[k] += fijk*(-R[j,k] + dot*rjk)
#                Eang += ka/2*(np.cos(Tijk)-0.5)**2
            '''
                Xijk = np.cross(rij, rjk)
                for l in diheds[nang]:
                    rkl = R[k,l]/L[k,l]
                    Xjkl = np.cross(rjk, rkl)
                    Tjkl = np.arccos(np.dot(rjk, rkl))
                    ratio = np.dot(Xijk,Xjkl)/np.sin(Tijk)/np.sin(Tjkl)
                    Edih += kd/2*(np.arccos(ratio)- dihed0[nang,ndih])**2
                    ndih+=1
                nang+=1
            Eb = Ebl + Eang + Edih
            '''
        elif self.mode != 'cluster':
            bonds, nobonds = self._get_bonds(pos)
            from constants import harmonic
            Rij = pos[bonds[:,0],:] - pos[bonds[:,1],:]
            rij = norm(Rij, axis=1)
            fij = harmonic['k']*(1 - harmonic['R0']/rij)
            for b in range(len(bonds)):
                Fb[bonds[b,0]] -= fij[b]*Rij[b,:]
                Fb[bonds[b,1]] += fij[b]*Rij[b,:]

#        elif self.mode !:
#            raise NotImplementedError('bondtype ' + self.bondtype + ' unknown')

        # repulsion
        Fr = np.zeros((len(pos),3))
        if self.reptype == 'expharm':
            from constants import rep_harmonic
            Rij = pos[nobonds[:,0],:]-pos[nobonds[:,1],:]
            r = norm(Rij, axis=1)
            fij = rep_harmonic['A']*rep_harmonic['b']*np.exp(-rep_harmonic['b']*r)/r
            for pair in range(len(nobonds)):
                Fr[nobonds[pair,0]] += fij[pair]*Rij[pair,:]
                Fr[nobonds[pair,1]] -= fij[pair]*Rij[pair,:]


        if self.mtd_refs:
            from utils.rmsd import rmsd_qcp as rmsd
            raise NotImplementedError('metadynamics not yet implemented')

        return Fb + Fr #+ Fmtd
     



