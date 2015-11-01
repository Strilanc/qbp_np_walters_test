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


def matrix_lift(f):
    """
    Lifts a function to apply to a matrix. Works by taking the spectral
    decomposition of the matrix, then applying the function to the eigenvalue
    weights, then summing the matrix pieces back together with the new weights.
    :param f: The number->number function to lift to a matrix->matrix function.
    >>> sqrtm = matrix_lift(cmath.sqrt)
    >>> np.allclose(sqrtm(X), np.mat([[1, -1j], [-1j, 1]]) * (1+1j)/2)
    True
    """

    def matrix_lift_helper(m, f):
        w, v = np.linalg.eig(m)
        result = np.mat(np.zeros(m.shape, np.complex128))
        for i in range(len(w)):
            eigen_val = w[i]
            eigen_vec = np.mat(v[:, i])
            eigen_mat = np.dot(eigen_vec, eigen_vec.H)
            result += f(eigen_val) * eigen_mat
        return result

    return lambda m: matrix_lift_helper(m, f)




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
    >>> np.allclose(op_pow(Z, 0.5), [[1, 0], [0, 1j]])
    True
    >>> np.allclose(op_pow(Z, -0.5), [[1, 0], [0, -1j]])
    True
    """
    return op_op(op, lambda e: e**p)


def bloch_rot(op, θ):
    """
    Returns the standard 'rotate around bloch-sphere axis' operation, based on
    exponentiating the single-qubit operation.

    >>> np.allclose(bloch_rot(Y, π/2), np.mat([[1,-1],[1,1]])/math.sqrt(2))
    True
    >>> np.allclose(bloch_rot(Y, -π/2), np.mat([[1,1],[-1,1]])/math.sqrt(2))
    True
    >>> np.allclose(bloch_rot(Y, π), [[0,-1],[1,0]])
    True
    >>> np.allclose(bloch_rot(Z, π), [[-1j,0],[0,1j]])
    True
    >>> np.allclose(bloch_rot(X, τ), [[-1,0],[0,-1]])
    True
    >>> np.allclose(bloch_rot(Y, τ), [[-1,0],[0,-1]])
    True
    >>> np.allclose(bloch_rot(Z, τ), [[-1,0],[0,-1]])
    True
    """
    return op_op(op, lambda e: cmath.exp(-1j * e * θ / 2))


class Context:
    def __init__(self, qubit_count):
        self.qubit_count = qubit_count

    def expand_op(self, qubit_op_matrix, qubit_index):
        """
        >>> np.allclose(Context(1).expand_op(X, 0), [[0,1],[1,0]])
        True
        >>> np.allclose(Context(2).expand_op(Z, 0), [[1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,-1]])
        True
        >>> np.allclose(Context(2).expand_op(Z, 1), [[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,-1]])
        True
        """
        post = np.identity(1 << qubit_index)
        pre = np.identity(1 << (self.qubit_count - qubit_index - 1))
        return np.kron(np.kron(pre, qubit_op_matrix), post)

    def σx(self, n):
        """
        Returns a matrix that applies a Y gate to the n'th qubit.
        :param n: Index of the qubit.
        """
        return self.expand_op(X, n)

    def σy(self, n):
        """
        Returns a matrix that applies a Y gate to the n'th qubit.
        :param n: Index of the qubit.
        """
        return self.expand_op(Y, n)

    def σz(self, n):
        """
        Returns a matrix that applies a Y gate to the n'th qubit.
        :param n: Index of the qubit.
        """
        return self.expand_op(Z, n)

    def R(self, n, θ):
        """
        Applies a θ radian rotation around the Y axis to the n'th qubit.
        :param n: The index of the qubit to rotate.
        :param θ: The amount to rotate.
        >>> s = math.sqrt(0.5)
        >>> np.allclose(Context(1).R(0, π/2), [[s,s],[-s,s]])
        True
        >>> np.allclose(Context(2).R(0, π/2), [[s,s,0,0],[-s,s,0,0],[0,0,s,s],[0,0,-s,s]])
        True
        >>> np.allclose(Context(2).R(1, π/2), [[s,0,s,0],[0,s,0,s],[-s,0,s,0],[0,-s,0,s]])
        True
        """
        return matrix_lift(lambda λ: cmath.exp(λ * 1j * θ / 2))(self.σy(n))


    def Rc(self, n, m, i, θ, on):
        """
        :param n: Index of the control qubit.
        :param m: Index of the target qubit.
        :param i: Control axis.
        :param θ: The amount to rotate.
        :param on: Whether the control should be on vs off for things to happen.
        """
        f = matrix_lift(lambda λ: cmath.exp((-1 if on else 1) * λ * 1j * θ / 2))
        c = self.σy(m)*self.expand_op(i, n)
        return f(c)

    def CROT(self, c, t, i, θ):
        """
        Controlled rotation of the t'th qubit around the Y axis, conditioned on
        the c'th qubit along the given i axis.
        :param c: Index of the control qubit.
        :param t: Index of the target qubit.
        :param i: The control axis.
        :param θ: The amount to rotate by.
        >>> s = math.sqrt(0.5)
        >>> np.allclose(\
                Context(2).CROT(1, 0, Z, π/2), \
                [[1,0,0,0],\
                 [0,1,0,0],\
                 [0,0,s,s],\
                 [0,0,-s,s]])
        True
        >>> np.allclose(\
                Context(2).CROT(0, 1, Z, π/2), \
                [[1,0,0,0],\
                 [0,s,0,s],\
                 [0,0,1,0],\
                 [0,-s,0,s]])
        True
        >>> np.allclose(\
                np.dot(Context(2).expand_op(H, 0), np.dot(Context(2).CROT(0, 1, X, π/2), Context(2).expand_op(H, 0))), \
                [[1,0,0,0],\
                 [0,s,0,s],\
                 [0,0,1,0],\
                 [0,-s,0,s]])
        True
        """
        return self.R(t, θ/2) * self.Rc(c, t, i, θ/2, True)

    def SCROTz(self, c, θ, s):
        """
        :param c: The target qubit.
        :param θ: Amount to rotate by.
        :param s: Index of a scratch qubit (initialized to zero).
        """
        Ps = self.expand_op(I, s)
        return Ps * self.Rc(s, c, X, θ, True) * self.Rc(c, s, Z, π/2, True)

    def SCROTx(self, c, θ, s):
        """
        :param c: The target qubit.
        :param θ: Amount to rotate by.
        :param s: Index of a scratch qubit (initialized to zero).
        """
        # Note: paper says the axis is `i = cos(φ) X + sin(φ) Z`
        # but I think the X and Z were reversed by accident, since otherwise
        # SCROTx would just be SCROTz
        return self.R(c, π/2) * self.SCROTz(c, θ, s) * self.R(c, -π/2)

    def QOR(self, c1, c2, s, s2):
        return self.R(s, π) \
            * self.SCROTx(s, π/2, s2) \
            * self.CROT(c2, s, Z, π/2) \
            * self.CROT(c1, s, Z, π/2)
