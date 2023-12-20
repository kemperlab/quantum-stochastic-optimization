from typing import Literal
from jax import Array

import pennylane as qml


def hamiltonian_ansatz(
    parameters: Array,
    single_gate: Literal['x', 'y', 'z'],
    double_gate: Literal['x', 'y', 'z'],
    n_var: int,
) -> qml.Hamiltonian:
    terms_single = []
    terms_double = []

    single_op = qml.PauliX if single_gate == 'x' else qml.PauliY if single_gate == 'y' else qml.PauliZ
    double_op = qml.PauliX if double_gate == 'x' else qml.PauliY if double_gate == 'y' else qml.PauliZ

    for i in range(n_var):
        terms_single.append(single_op(i))

        if i >= 1:
            terms_double.append(double_op(i) @ double_op(i - 1))

    hamiltonian = (qml.Hamiltonian(parameters[:n_var], terms_single) +
                   qml.Hamiltonian(parameters[n_var:], terms_double))

    hamiltonian.compute_grouping('commuting')
    return hamiltonian
