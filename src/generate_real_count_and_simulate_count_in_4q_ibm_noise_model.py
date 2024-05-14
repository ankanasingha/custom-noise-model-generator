from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit, transpile
import numpy as np
from qiskit_aer.noise import NoiseModel
from qiskit_aer import Aer
import os

np.random.seed(42)


def create_random_circuits(selected_qubits, depth, ecr_probability):
    num_qubits = len(selected_qubits)
    circuit = QuantumCircuit(max(selected_qubits) + 1)
    for _ in range(depth):
        if np.random.random() < ecr_probability:
            qubit_idx, target_idx = np.random.choice(selected_qubits, size=2, replace=False)
            circuit.ecr(qubit_idx, target_idx)
        else:
            qubit_idx = np.random.choice(selected_qubits)
            angle = np.random.uniform(0, 2 * np.pi)
            circuit.rz(angle, qubit_idx)

    for qubit in selected_qubits:
        circuit.sx(qubit)

    circuit.measure_all()

    return circuit


def login_to_ibm_quantum():
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
        token=os.environ['IBM_TOKEN']
    )
    return service.backend('ibm_osaka')


def generate_random_circuit():
    backend = login_to_ibm_quantum()
    selected_qubits = [0, 1, 2, 3]
    num_circuits = 100
    depth = 20
    ecr_probability = 0.80

    random_circuits = [create_random_circuits(selected_qubits, depth, ecr_probability) for _ in range(num_circuits)]

    transpiled_circuits = transpile(random_circuits, backend=backend, initial_layout=selected_qubits,
                                    optimization_level=0)
    real_job = backend.run(transpiled_circuits, shots=8192)

    real_results = real_job.result()

    real_counts = [real_results.get_counts(i) for i in range(num_circuits)]

    noise_model = NoiseModel.from_backend(backend)

    simulator = Aer.get_backend('qasm_simulator')
    transpiled_simulated_circuits = transpile(random_circuits, backend=backend, initial_layout=selected_qubits,
                                              optimization_level=3)
    simulated_job = simulator.run(transpiled_simulated_circuits,
                                    noise_model=noise_model,
                                    shots=8192)

    simulated_results = simulated_job.result()

    simulated_counts = [simulated_results.get_counts(i) for i in range(num_circuits)]
    print('Real Counts:',real_counts)
    print('Simulated Counts Using IBM noise Model:',simulated_counts)


if __name__ == "__main__":
    generate_random_circuit()
