# -*- coding: utf-8 -*-
from density_matrix import DensityMatrix
import numpy as np
import math
import cmath


np.set_printoptions(precision=5, suppress=True)

π = math.pi
τ = π*2
I = np.matrix([[1, 0], [0, 1]])
X = np.matrix([[0, 1], [1, 0]])
Y = np.matrix([[0, -1j], [1j, 0]])
Z = np.matrix([[1, 0], [0, -1]])
H = np.matrix([[1, 1], [1, -1]])/math.sqrt(2)


def op_op(op, f):
    """
    Applies an analytic operation to a single-qubit operation's matrix.
    """
    a, b = f(1), f(-1)
    s, d = (a+b)/2, (a-b)/2
    return I*s + op*d


def op_pow(op, p):
    """
    Raises a single-qubit operation to a fractional power (e.g. to compute
    square roots use p=0.5).

    >>> np.allclose(op_pow(X, 0.5), np.mat([[1, -1j], [-1j, 1]]) * (1+1j)/2)
    True
    >>> np.allclose(op_pow(X, -0.5), np.mat([[1, 1j], [1j, 1]]) * (1-1j)/2)
    True
    >>> np.allclose(op_pow(Z, 0.5), np.mat([[1, 0], [0, 1j]]))
    True
    >>> np.allclose(op_pow(Z, -0.5), np.mat([[1, 0], [0, -1j]]))
    True
    """
    return op_op(op, lambda e: e**p)


def R(op, theta):
    """
    Returns the standard 'rotate around bloch-sphere axis' operation, based on
    exponentiating the single-qubit operation.

    >>> np.allclose(R(Y, π/2), np.mat([[1,-1],[1,1]])/math.sqrt(2))
    True
    >>> np.allclose(R(Y, -π/2), np.mat([[1,1],[-1,1]])/math.sqrt(2))
    True
    >>> np.allclose(R(Y, π), np.mat([[0,-1],[1,0]]))
    True
    >>> np.allclose(R(Z, π), np.mat([[-1j,0],[0,1j]]))
    True
    >>> np.allclose(R(X, τ), np.mat([[-1,0],[0,-1]]))
    True
    >>> np.allclose(R(Y, τ), np.mat([[-1,0],[0,-1]]))
    True
    >>> np.allclose(R(Z, τ), np.mat([[-1,0],[0,-1]]))
    True
    """
    return op_op(op, lambda e: cmath.exp(-1j * e * theta / 2))


