from dataclasses import dataclass
from typing import Union, List, Dict
import numpy as np
from numpy.typing import NDArray

# Abstract base class
class State:
    """Abstract base for a quantum state."""
    def evolve(self, hamiltonian, time):
        """Evolve state according to Schrödinger equation."""
        raise NotImplementedError

    def expectation(self, observable):
        """Calculate expectation value of an observable."""
        raise NotImplementedError

@dataclass
class PureState(State):
    """Represents a pure state as a vector in Hilbert space."""
    data: NDArray[np.complex128]
    basis: List[str]

    def evolve(self, hamiltonian, time):
        """Evolve by applying unitary operator U = exp(-i*H*t)."""
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        U = eigenvectors @ np.diag(np.exp(-1j * eigenvalues * time)) @ eigenvectors.conj().T
        self.data = U @ self.data

    def normalize(self) -> None:
        norm = np.linalg.norm(self.data)
        self.data /= norm

    @property
    def normalized(self) -> 'PureState':
        new_state = self.copy()
        new_state.normalize()
        return new_state


    def expectation(
        self,
        observable
    ) -> np.float64:
        """⟨ψ|A|ψ⟩"""
        return np.float64(np.vdot(self.data, observable @ self.data).real)
    
    def copy(self) -> 'PureState':
        return PureState(self.data.copy(), self.basis.copy())

@dataclass
class MixedState(State):
    """Represents a mixed state as a density matrix."""
    density_matrix: NDArray[np.complex128]

    def evolve(self, hamiltonian, time):
        """Evolve density matrix: ρ(t) = U ρ(0) U†."""
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
        U = eigenvectors @ np.diag(np.exp(-1j * eigenvalues * time)) @ eigenvectors.conj().T
        self.density_matrix = U @ self.density_matrix @ U.conj().T

    def expectation(self, observable):
        """⟨A⟩ = Tr(ρ A)"""
        return np.trace(self.density_matrix @ observable).real

    @classmethod
    def from_ensemble(cls, states_and_probs: Dict[np.complex128, PureState]):
        """Create a mixed state from an ensemble of pure states.
        This illustrates that the density matrix is the fundamental representation."""
        dim = 0
        rho = np.zeros((dim, dim), dtype=np.complex128)
        total_prob = 0.0
        for prob, state in states_and_probs.items():
            # Ensure state is a PureState with a vector
            rho += prob * np.outer(state.data.conj(), state.data)
            total_prob += prob
        # Optionally renormalize probabilities if they don't sum to 1
        if not np.isclose(total_prob, 1.0):
            rho = rho / total_prob
        return cls(density_matrix=rho)