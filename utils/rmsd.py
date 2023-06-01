"""
RMSD calculation, via quaterion-based characteristic polynomial in
pure python/numpy.

Reference
---------
Rapid calculation of RMSDs using a quaternion-based characteristic polynomial.
Acta Crystallogr A 61(4):478-480.

Originally downloaded from https://gist.github.com/rmcgibbo/5522602 
Adapted by Mario Galante (mario.galante@uni.lu), October 2021

"""
import numpy as np

def _center(conformation):
    """Center and typecheck the conformation"""

    conformation = np.asarray(conformation)
    if not conformation.ndim == 2:
        raise ValueError('conformation must be two dimensional')
    _, three = conformation.shape
    if not three == 3:
        raise ValueError('conformation second dimension must be 3')

    centroid = np.mean(conformation, axis=0)
    centered = conformation - centroid
    return centered


def rmsd_qcp(conformation1, conformation2):
    """Compute the RMSD with Theobald's quaterion-based characteristic
    polynomial
    
    Rapid calculation of RMSDs using a quaternion-based characteristic polynomial.
    Acta Crystallogr A 61(4):478-480.
    
    Parameters
    ----------
    conformation1 : np.ndarray, shape=(n_atoms, 3)
        The cartesian coordinates of the first conformation
    conformation2 : np.ndarray, shape=(n_atoms, 3)
        The cartesian coordinates of the second conformation

    Returns
    -------
    rmsd : float
        The root-mean square deviation after alignment between the two pointsets
    """
    # center and typecheck the conformations
    A = _center(conformation1)
    B = _center(conformation2)
    if not A.shape[0] == B.shape[0]:
        raise ValueError('conformation1 and conformation2 must have same number of atoms')
    n_atoms = len(A)

    #the inner product of the structures A and B
    G_A = np.einsum('ij,ij', A, A)
    G_B = np.einsum('ij,ij', B, B)
    #print 'GA', G_A, np.trace(np.dot(A.T, A))
    #print 'GB', G_B, np.trace(np.dot(B.T, B))

    # M is the inner product of the matrices A and B
    M = np.dot(B.T, A)

    # unpack the elements
    Sxx, Sxy, Sxz = M[0, :]
    Syx, Syy, Syz = M[1, :]
    Szx, Szy, Szz = M[2, :]

    # do some intermediate computations to assemble the characteristic
    # polynomial
    Sxx2 = Sxx * Sxx
    Syy2 = Syy * Syy
    Szz2 = Szz * Szz

    Sxy2 = Sxy * Sxy
    Syz2 = Syz * Syz
    Sxz2 = Sxz * Sxz

    Syx2 = Syx * Syx
    Szy2 = Szy * Szy
    Szx2 = Szx * Szx

    SyzSzymSyySzz2 = 2.0*(Syz*Szy - Syy*Szz)
    Sxx2Syy2Szz2Syz2Szy2 = Syy2 + Szz2 - Sxx2 + Syz2 + Szy2

    # two of the coefficients
    C2 = -2.0 * (Sxx2 + Syy2 + Szz2 + Sxy2 + Syx2 + Sxz2 + Szx2 + Syz2 + Szy2)
    C1 = 8.0 * (Sxx*Syz*Szy + Syy*Szx*Sxz + Szz*Sxy*Syx - Sxx*Syy*Szz - Syz*Szx*Sxy - Szy*Syx*Sxz)

    SxzpSzx = Sxz + Szx
    SyzpSzy = Syz + Szy
    SxypSyx = Sxy + Syx
    SyzmSzy = Syz - Szy
    SxzmSzx = Sxz - Szx
    SxymSyx = Sxy - Syx
    SxxpSyy = Sxx + Syy
    SxxmSyy = Sxx - Syy
    Sxy2Sxz2Syx2Szx2 = Sxy2 + Sxz2 - Syx2 - Szx2

    # the other coefficient
    C0 = Sxy2Sxz2Syx2Szx2 * Sxy2Sxz2Syx2Szx2 \
        + (Sxx2Syy2Szz2Syz2Szy2 + SyzSzymSyySzz2) * (Sxx2Syy2Szz2Syz2Szy2 - SyzSzymSyySzz2) \
        + (-(SxzpSzx)*(SyzmSzy)+(SxymSyx)*(SxxmSyy-Szz)) * (-(SxzmSzx)*(SyzpSzy)+(SxymSyx)*(SxxmSyy+Szz)) \
        + (-(SxzpSzx)*(SyzpSzy)-(SxypSyx)*(SxxpSyy-Szz)) * (-(SxzmSzx)*(SyzmSzy)-(SxypSyx)*(SxxpSyy+Szz)) \
        + (+(SxypSyx)*(SyzpSzy)+(SxzpSzx)*(SxxmSyy+Szz)) * (-(SxymSyx)*(SyzmSzy)+(SxzpSzx)*(SxxpSyy+Szz)) \
        + (+(SxypSyx)*(SyzmSzy)+(SxzmSzx)*(SxxmSyy-Szz)) * (-(SxymSyx)*(SyzpSzy)+(SxzmSzx)*(SxxpSyy-Szz))

    # Netwtorn-Raphson
    E0 = (G_A + G_B) / 2.0
    max_eigenvalue = E0
    for i in range(50):
        old_g = max_eigenvalue
        x2 = max_eigenvalue * max_eigenvalue
        b = (x2 + C2) * max_eigenvalue
        a = b + C1
        delta = ((a * max_eigenvalue + C0)/(2.0 * x2 * max_eigenvalue + b + a))
        max_eigenvalue -= delta
        if abs(max_eigenvalue - old_g) < abs(1e-11 * max_eigenvalue):
            break
    if i >= 50:
        raise ValueError('More than 50 iterations needed.')

    rmsd = np.sqrt(np.abs(2.0 * (E0 - max_eigenvalue) / n_atoms))
    return rmsd


def rmsd_msmb(conf1, conf2):
    from msmbuilder.metrics.rmsd import RMSD
    rmsd = RMSD()
    A = rmsd.prepare_trajectory({'XYZList': np.array([conf1])})
    B = rmsd.prepare_trajectory({'XYZList': np.array([conf2])})
    return rmsd.one_to_all(A, B, 0)[0]


def test():
    A = np.arange(120).reshape(40,3) / 50.0
    B = np.arange(120).reshape(40,3) / 50.0
    A[2,0] += .1

    np.testing.assert_approx_equal(rmsd_qcp(A, B), rmsd_msmb(A, B), significant=4)
    print('passed')

if __name__ == '__main__':
    test()
