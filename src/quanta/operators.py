import numpy as np
from .mathematics import i

pauli_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
pauli_y = np.array([[0, -i], [i, 0]], dtype=np.complex128)
pauli_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)