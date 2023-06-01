import numpy as np
from ase.io.trajectory import TrajectoryReader
import matplotlib.pyplot as plt

def linear_density(traj):
    print(len(traj))
    pos0 = traj[0].get_positions()
    Nat = len(pos0)
    L = float(pos0[-1,0]-pos0[0,0])
    f = 1.52*(Nat-1)/L
    nbins = int(Nat)
    dbins = L/nbins
    bins = np.linspace(0, nbins, num=nbins)
    print(bins)
    D = np.zeros(int(nbins))
    ats = np.linspace(0,Nat,num=Nat)
    print(nbins)
    ts = [int(i*8e4) for i in range(5)]
    for t in ts:
        pos = traj[t].get_positions()
        for x in pos[:-1,0]:
            bin = int(x/dbins)
#        print(x, x/dbins)
            D[bin]+=1
        #plt.plot(bins,D, label=t)
        plt.plot(ats,pos[:,1], label=t)
    plt.legend()
    plt.show()



def get_xvariance(traj, t0, tf, step):
    pos0 = traj[0].get_positions()
    Nat = len(pos0)
    L = float(pos0[-1,0]-pos0[0,0])
    f = 1.52*(Nat-1)/L
    Nit = len(traj)
    times = np.linspace(t0, tf, num=int((tf-t0)/step))#1000)
    avs, sigmas = [], []
    for t in times:
        pos = traj[int(t)].get_positions()
        av = np.sum(pos[:,0])/Nat
        avs.append(av)
        dav2 = (pos[:,0]-av)**2
        sigma = np.sqrt(np.sum(dav2))/Nat
        sigmas.append(np.sqrt(np.sum(dav2))/Nat)


    return times, np.asarray(sigmas)

   
#mbd = TrajectoryReader('/home/mario/Research/Postdoc/data/chain/Paul_params/dynamics/40n5f2/mbd/chainGS-MBD.traj')
#ts = TrajectoryReader('/home/mario/Research/Postdoc/data/chain/Paul_params/dynamics/40n5f2/ts/chainGS-TS.traj')
t1 = TrajectoryReader('/Users/mario/Work/Research/Postdoc/data/chain/dynamics/40n5f2/MBD/5K/1/chain.traj')
t3 = TrajectoryReader('/Users/mario/Work/Research/Postdoc/data/chain/dynamics/40n5f2/MBD/5K/3/chain.traj')

log1 = np.loadtxt('/Users/mario/Work/Research/Postdoc/data/chain/dynamics/40n5f2/MBD/5K/1/chain.log')
log3 = np.loadtxt('/Users/mario/Work/Research/Postdoc/data/chain/dynamics/40n5f2/MBD/5K/3/chain.log')

#linear_density(traj)
s1 = get_xvariance(t1, 100, 20000, 100)
s3 = get_xvariance(t3, 100, 20000, 100)


plt.scatter(s1[0]*2e-3, s1[1], label = '1')
plt.scatter(s3[0]*2e-3, s3[1], label = '3')
plt.xlabel('time [ps]')
plt.ylabel('var(R_x)')
plt.legend()
plt.figure()
plt.plot(log1[3000:, 0], log1[3000:,1])
plt.plot(log3[3000:, 0], log3[3000:,1])
plt.show()
#plt.savefig('fig_variance.pdf')
        
