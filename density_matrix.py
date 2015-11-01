# -*- coding: utf-8 -*-
import numpy as np
import functools
import math


class DensityMatrix:
    """
    A representation of mixed quantum states.
    """

    def __init__(self, state_matrix):
        self.state = state_matrix

    @staticmethod
    def zeros(bit_count):
        ρ = np.zeros((1 << bit_count, 1 << bit_count))
        ρ[0, 0] = 1
        return DensityMatrix(ρ)

    def full_op(self, op_matrix):
        """
        Applies an operation to the state and returns the resulting state.
        :param: op_matrix A unitary matrix corresponding to the operation.
        :return: The post-operation state (as a density matrix).
        """
        # ρ₂ = U ρ U⁻¹
        m = np.dot(np.dot(op_matrix, self.state), op_matrix.H)
        return DensityMatrix(m)

    def qubit_op(self, op_matrix, qubit_index):
        """
        Applies an operation to one of the qubits in the state, and returns
        the resulting state.
        :param: op_matrix A unitary matrix corresponding to the qubit operation.
        :param: qubit_index The qubit to apply the operation to.
        :return: The post-operation state (as a density matrix).

        >>> off = DensityMatrix(np.mat([[1, 0], [0, 0]]))
        >>> on = DensityMatrix(np.mat([[0, 0], [0, 1]]))
        >>> X = np.mat([[0, 1], [1, 0]])
        >>> H = np.mat([[1, 1], [1, -1]])/math.sqrt(2)
        >>> offoff = DensityMatrix(np.mat([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]))
        >>> onoff = DensityMatrix(np.mat([[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,0]]))
        >>> offon = DensityMatrix(np.mat([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,0]]))
        >>> np.allclose(off.qubit_op(X, 0).state, on.state)
        True
        >>> np.allclose(on.qubit_op(X, 0).state, off.state)
        True
        >>> np.allclose(off.qubit_op(H, 0).state, np.mat([[0.5, 0.5], [0.5, 0.5]]))
        True
        >>> np.allclose(on.qubit_op(H, 0).state, np.mat([[0.5, -0.5], [-0.5, 0.5]]))
        True
        >>> np.allclose(offoff.qubit_op(X, 0).state, onoff.state)
        True
        >>> np.allclose(offoff.qubit_op(X, 1).state, offon.state)
        True
        """

        # Use tensor product to expand operation to apply to full state.
        for i in range(qubit_index):
            op_matrix = np.kron(op_matrix, np.identity(2))
        bit_count = int(round(math.log2(self.state.shape[0])))
        for i in range(bit_count - qubit_index - 1):
            op_matrix = np.kron(np.identity(2), op_matrix)

        return self.full_op(op_matrix)

    def measure_bit(self, bit_index):
        """
        Determines the expected results of measuring the given qubit.

        :param bit_index: The index of the qubit to measure (along the Z axis).
        :return: A list of (result, probability, state) outcome tuples.
        """
        n = self.state.shape[0]
        projector_on = np.zeros((n, n))
        projector_off = np.zeros((n, n))
        for i in range(n):
            target = projector_on if i & (1 << bit_index) else projector_off
            target[i, i] = 1

        off_state = np.dot(np.dot(projector_off, self.state), projector_off)
        on_state = np.dot(np.dot(projector_on, self.state), projector_on)

        off_p = off_state.trace()[0, 0]
        on_p = on_state.trace()[0, 0]

        results = []
        if off_p > 0.00001:
            results.append((False, off_p, DensityMatrix(off_state / off_p)))
        if on_p > 0.00001:
            results.append((True, on_p, DensityMatrix(on_state / on_p)))
        return results

    def measure_bit_ignore_result(self, bit_index):
        """
        Returns the state corresponding to what is known about the state after
        a qubit has been measured, but without conditioning on the measurement
        result (so the various possibilities just get added together).
        :param bit_index: The qubit to measure.
        :return: A density matrix for what is known about the post-measurement
        state.

        >>> hOff = DensityMatrix(np.mat([[1,1],[1,1]])/2)
        >>> hOn = DensityMatrix(np.mat([[1,-1],[-1,1]])/2)
        >>> coin = DensityMatrix(np.mat([[0.5,0],[0,0.5]]))
        >>> np.allclose(hOff.measure_bit_ignore_result(0).state, coin.state)
        True
        >>> np.allclose(hOn.measure_bit_ignore_result(0).state, coin.state)
        True
        """
        combined = functools.reduce(
            lambda a, e: a + e[1] * e[2].state,
            self.measure_bit(bit_index),
            np.zeros(self.state.shape))
        return DensityMatrix(combined)

    def __str__(self):
        return str(self.state)

    def spectral_decomposition_str(self):
        w, v = np.linalg.eig(self.state)
        combo = []
        for i in range(len(w)):
            combo.append((w[i], v[:, i].transpose()))
        kept = [e for e in combo if abs(e[0]) > 0.00001]
        desc = ["(" + str(e[0]) + ")*" + str(e[1]) + "^2" for e in kept]
        return " + ".join(desc)

    def __repr__(self):
        return "DensityMatrix(" + repr(self.state) + ")"
