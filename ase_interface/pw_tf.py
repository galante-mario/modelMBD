from __future__ import division, print_function
from math import pi, inf
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.python import debug as tf_debug
tf.disable_v2_behavior()

inf = float('inf')
pi = tf.constant(pi, tf.float64)
inf = tf.constant(inf, tf.float64)
#ang = 1/0.529177249
ang = 1.889725989

class PWEvaluator(object):
    def __init__(self, hessians=False, **kwargs):
        self._inputs = coords, eps, sigma, R_vdw = [
            tf.placeholder(tf.float64, shape=shape, name=name)
            for shape, name in [
                ((None, 3), 'coords'),
                ((None, ), 'eps'),
                ((None, ), 'sigma'),
                ((None, ), 'R_vdw')
            ]
        ]
        self._output = pw_energy(*self._inputs, **kwargs)
        self._grad = tf.gradients(self._output, [self._inputs[0]])[0]

        if hessians:
            self._init_hessians()
        else:
            self._hessians = None

    def _init_hessians(self):
        self._hessians = tf.hessians(self._output, [self._inputs[0]])[0]


    def __call__(self, coords, eps, sigma, R_vdw, hessians=None):
        inputs = dict(zip(self._inputs, [coords, eps, sigma, R_vdw]))
        outputs = self._output, self._grad
        if hessians or hessians is None and self._hessians is not None:
            if self._hessians is None:
                self._init_hessians()
            outputs = self._hessians
        with tf.Session() as sess:
            out = sess.run(outputs, inputs)
            return out

def damping_fermi(R, S_vdw, d):
    return 1/(1+tf.exp(-d*(R/S_vdw/0.94-1)))
def damping_f(R, S_vdw):
    return (1-tf.exp(-4.0*(R/S_vdw)**1))**7

def T_bare(R, alpha_0, C6):
    a = _set_diag(tf.reduce_sum(R**2, -1), 1e-10)
    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(R**2, -1), 1e-10))
    R_6 = _set_diag(R_1**6, inf)
    C6_ij = tf.sqrt((C6[:, None]*C6[None, :]))
    return -C6_ij[:,:,None,None]/R_6[:,:,None,None], 1/R_6
'''
def T_bare(R, eps, sigma):
    R_1 = tf.sqrt(_set_diag(tf.reduce_sum(R**2, -1), 1e-10))
    R_6 = _set_diag(R_1**6, inf)
    epsij   = tf.sqrt((eps[:, None]*eps[None, :]))
    sigmaij = 0.5*(sigma[:, None]+sigma[None, :])
    return -4*epsij[:,:,None,None]*sigmaij[:,:,None,None]**6/R_6[:,:,None,None]
'''
def _set_diag(A, val):
    return tf.matrix_set_diag(A, tf.fill(tf.shape(A)[0:1], tf.cast(val, tf.float64)))

def _repeat(a, n):
    return tf.reshape(tf.tile(a[:, None], (1, n)), (-1,))

def pw_energy(coords, alpha_0, C6, R_vdw):
    Rs = coords[:, None, :]-coords[None, :, :]
    dists = tf.sqrt(_set_diag(tf.reduce_sum(Rs**2, -1), 1e-10))
    S_vdw = 0.5*(R_vdw[:, None]+R_vdw[None, :])
    tbare, other = T_bare(Rs, alpha_0, C6)
#    ene  = damping_fermi(dists, S_vdw, 6.0)[:, :, None, None]*T_bare(Rs, alpha_0, C6)
    df = damping_fermi(dists, S_vdw, 6.0)
    ene  = damping_fermi(dists, S_vdw, 6.0)[:, :, None, None]*tbare
    return tf.reduce_sum(ene)/2, other
'''
def pw_energy(coords, eps, sigma, R_vdw):
    Rs = coords[:, None, :]-coords[None, :, :]
    dists = tf.sqrt(_set_diag(tf.reduce_sum(Rs**2, -1), 1e10))
    S_vdw = 0.5*(R_vdw[:, None]+R_vdw[None, :])
    ene  = damping_fermi(dists, S_vdw, 6.0)[:, :, None, None]*T_bare(Rs, eps, sigma)
    return tf.reduce_sum(ene)/2 
'''

