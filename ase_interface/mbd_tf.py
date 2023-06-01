from __future__ import division, print_function
import sys
from math import pi, inf
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python import debug as tf_debug
tf.disable_v2_behavior()

inf = float('inf')
pi = tf.constant(pi, tf.float64)
inf = tf.constant(inf, tf.float64)
ang = 1/0.529177249

class MBDEvaluator(object):
    def __init__(self, hessians=False, **kwargs):
        self._inputs = coords, alpha_0, C6, R_vdw, V = [
            tf.placeholder(tf.float64, shape=shape, name=name)
            for shape, name in [
                ((None, 3), 'coords'),
                ((None, ), 'alpha_0'),
                ((None, ), 'C6'),
                ((None, ), 'R_vdw'),
                ((None, ), 'V')
            ]
        ]
        self._output = mbd_energy(*self._inputs, **kwargs)
        self._grad = tf.gradients(self._output, [self._inputs[0]])[0]

        if hessians:
            self._init_hessians()
        else:
            self._hessians = None

    def _init_hessians(self):
        self._hessians = tf.hessians(self._output, [self._inputs[0]])[0]


    def __call__(self, coords, alpha_0, C6, R_vdw, V, hessians=None):
        inputs = dict(zip(self._inputs, [coords, alpha_0, C6, R_vdw, V]))
        outputs = self._output, self._grad
        if hessians or hessians is None and self._hessians is not None:
            if self._hessians is None:
                self._init_hessians()
            outputs = self._hessians
        with tf.Session() as sess:
            out = sess.run(outputs, inputs)
            return out

def damping_fermi(R, S_vdw, d):
    return 1/(1+tf.exp(-d*(R/S_vdw-1)))

def T_bare(R, R_vdw):
    R_2 = tf.reduce_sum(R**2, -1)
#    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(R**2, -1), 1e-10))
    R_1 = tf.sqrt(R_2)
    R_5 = _set_diag(R_1**5, inf)
    S_vdw = 0.81 * (R_vdw[:, None] + R_vdw[None, :])
    d = damping_fermi(R_1, S_vdw, 6.0)
    return (
            -3*d[:,:,None,None]*R[:, :, :, None]*R[:, :, None, :]
            + d[:,:,None,None]*R_2[:, :, None, None]*np.eye(3)[None, None, :, :]
            )/R_5[:,:,None,None]

def _set_diag(A, val):
    return tf.matrix_set_diag(A, tf.fill(tf.shape(A)[0:1], tf.cast(val, tf.float64)))

def _repeat(a, n):
    return tf.reshape(tf.tile(a[:, None], (1, n)), (-1,))

def mbd_energy(coords, alpha_0, C6, R_vdw, V):
    omega =  4/3*C6/alpha_0**2
    sigma = (tf.sqrt(2/pi)*(alpha_0*V)/3)**(1/3)
    dipmat = dipole_matrix(coords, R_vdw*V**(1/3), sigma)
    pre = _repeat(omega*tf.sqrt(alpha_0*V), 3)
    mat = tf.diag(_repeat(omega**2, 3))+pre[:, None]*pre[None, :]*dipmat
    eigs = tf.linalg.eigvalsh(mat)
    ene = tf.reduce_sum(tf.sqrt(tf.abs(eigs)))/2-3*tf.reduce_sum(omega)/2
    return ene

def dipole_matrix(coords, R_vdw=None, sigma=None):
    Rs = coords[:, None, :]-coords[None, :, :]
    dists = tf.sqrt(_set_diag(tf.reduce_sum(Rs**2, -1), 1e-10))
    sigmaij = tf.sqrt(sigma[:, None]**2+sigma[None, :]**2)
    dipmat = T_bare(Rs, R_vdw)
    n_atoms = tf.shape(coords)[0]
    return tf.reshape(tf.transpose(dipmat, (0, 2, 1, 3)), (3*n_atoms, 3*n_atoms))


def T_erf_coulomb(R, sigma):
    bare = T_bare(R)
    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(R**2, -1), 1e-10))
    R_5 = _set_diag(R_1**5, inf)
    RR_R5 = R[:, :, :, None]*R[:, :, None, :]/R_5[:, :, None, None]
    zeta = R_1/(sigma*1.8)
    theta = 2*zeta/tf.sqrt(pi)*tf.exp(-zeta**2)
    erf_theta = tf.erf(zeta) -theta
    return erf_theta[:, :, None, None]*bare + \
        (2*(zeta**2)*theta)[:, :, None, None]*RR_R5

