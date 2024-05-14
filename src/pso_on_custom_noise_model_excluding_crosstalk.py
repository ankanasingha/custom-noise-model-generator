from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit import QuantumCircuit
from qiskit import transpile
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from qiskit_aer import Aer
from pyswarms.single.global_best import GlobalBestPSO
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error
)

np.random.seed(42)


def login_to_ibm_quantum():
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
        token=os.environ['IBM_TOKEN']
    )
    return service.backend('ibm_osaka')


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


def calculate_tvd(simulated_counts, real_counts):
    sim_probs = {k: v / 8192 for k, v in simulated_counts.items()}
    real_probs = {k: v / 8192 for k, v in real_counts.items()}
    return 0.5 * sum(abs(sim_probs.get(k, 0) - real_probs.get(k, 0)) for k in set(sim_probs) | set(real_probs))


def create_noise_model(ecr_error_rates, qubit_noise_params):
    noise_model = NoiseModel()

    for i in selected_qubits:
        # Preparation errors
        prep_errors = pauli_error(
            [('X', qubit_noise_params[i]['prob_meas1_prep0']), ('I', 1 - qubit_noise_params[i]['prob_meas1_prep0'])])
        noise_model.add_quantum_error(prep_errors, 'reset', [i])

    for qubit, params in qubit_noise_params.items():
        sx_error = pauli_error([('X', params['sx_error_rate']), ('I', 1 - params['sx_error_rate'])])
        noise_model.add_quantum_error(sx_error, 'sx', [qubit])

    for pair, error_rate in ecr_error_rates.items():
        qubits = [int(q) for q in pair.split('_')]
        control, target = qubits[0], qubits[1]

        # Define two-qubit depolarizing error for the ecr gate
        ecr_error = depolarizing_error(error_rate, 2)
        noise_model.add_quantum_error(ecr_error, 'ecr', qubits)

    # Thermal relaxation and measurement errors
    for qubit, params in qubit_noise_params.items():
        t1 = params['t1']
        t2 = params['t2']

        # Correct T_2 if it exceeds 2 * T_1
        if t2 > 2 * t1:
            t2 = 2 * t1
            qubit_noise_params[qubit]['t2'] = t2

            thermal_error = thermal_relaxation_error(qubit_noise_params[qubit]['t1'], qubit_noise_params[qubit]['t2'],
                                                     660e-9)  # using 660ns as gate time
            noise_model.add_quantum_error(thermal_error, 'measure', [qubit])  # Apply to measurement

    for i in selected_qubits:
        readout_error = ReadoutError(
            [[1 - qubit_noise_params[i]['prob_meas1_prep0'], qubit_noise_params[i]['prob_meas1_prep0']],
             [qubit_noise_params[i]['prob_meas0_prep1'], 1 - qubit_noise_params[i]['prob_meas0_prep1']]])
        noise_model.add_readout_error(readout_error, [i])

    return noise_model


def objective_function(x):
    for params in x:
        prob_meas1_prep0_q0, prob_meas0_prep1_q0, sx_error_rate_q0, \
        prob_meas1_prep0_q1, prob_meas0_prep1_q1, sx_error_rate_q1, \
        prob_meas1_prep0_q2, prob_meas0_prep1_q2, sx_error_rate_q2, \
        prob_meas1_prep0_q3, prob_meas0_prep1_q3, sx_error_rate_q3, \
        ecr_error_rate_1_0, ecr_error_rate_2_3, ecr_error_rate_2_1, \
        t1_q0, t2_q0, t1_q1, t2_q1, t1_q2, t2_q2, t1_q3, t2_q3 = params

    selected_qubits = [0, 1, 2, 3]
    qubit_noise_params = {
        0: {'prob_meas1_prep0': prob_meas1_prep0_q0, 'prob_meas0_prep1': prob_meas0_prep1_q0,
            'sx_error_rate': sx_error_rate_q0, 't1': t1_q0, 't2': t2_q0},
        1: {'prob_meas1_prep0': prob_meas1_prep0_q1, 'prob_meas0_prep1': prob_meas0_prep1_q1,
            'sx_error_rate': sx_error_rate_q1, 't1': t1_q1, 't2': t2_q1},
        2: {'prob_meas1_prep0': prob_meas1_prep0_q2, 'prob_meas0_prep1': prob_meas0_prep1_q2,
            'sx_error_rate': sx_error_rate_q2, 't1': t1_q2, 't2': t2_q2},
        3: {'prob_meas1_prep0': prob_meas1_prep0_q3, 'prob_meas0_prep1': prob_meas0_prep1_q3,
            'sx_error_rate': sx_error_rate_q3, 't1': t1_q3, 't2': t2_q3}
    }
    ecr_error_rates = {
        '1_0': ecr_error_rate_1_0,
        '2_3': ecr_error_rate_2_3,
        '2_1': ecr_error_rate_2_1
    }

    avg_tvd_by_iteration = []
    final_avg_tvd = 0
    initial_noise_model = create_noise_model(ecr_error_rates, qubit_noise_params)

    for i in range(1, 6):
        simulator = Aer.get_backend('qasm_simulator')
        simulated_job = simulator.run(transpiled_circuits, noise_model=initial_noise_model, shots=8192,
                                      seed_simulator=42, optimization_level=0)

        simulated_results = simulated_job.result()
        simulated_counts = [simulated_results.get_counts(j) for j in range(100)]

        real_counts_file_path = os.path.join(os.path.dirname(os.getcwd()), "resources", "4q", "real_counts",
                                             f"real_counts_{i}.json")
        with open(real_counts_file_path, 'r') as file:
            real_counts = json.load(file)
        avg_tvd = np.mean([calculate_tvd(simulated_counts[i], real_counts[i]) for i in range(len(real_counts))])
        avg_tvd_by_iteration.append(avg_tvd)
    final_avg_tvd = sum(avg_tvd_by_iteration) / 5
    return final_avg_tvd


if __name__ == "__main__":
    backend = login_to_ibm_quantum()
    selected_qubits = [0, 1, 2, 3]
    num_circuits = 100
    depth = 20
    ecr_probability = 0.8

    random_circuits = [create_random_circuits(selected_qubits, depth, ecr_probability) for _ in range(num_circuits)]

    transpiled_circuits = transpile(random_circuits,
                                    backend=backend,
                                    initial_layout=selected_qubits,
                                    optimization_level=0)

    lower_bounds = [
        1e-8, 1e-8, 1e-6,  # Noise for q0
        1e-8, 1e-8, 1e-6,  # Noise for q1
        1e-8, 1e-8, 1e-6,  # Noise for q2
        1e-8, 1e-8, 1e-6,  # Noise for q3
        1e-5, 1e-5, 1e-5,  # ECR error rates
        5e-5, 5e-5,  # T1 and T2 for q0
        5e-5, 5e-5,  # T1 and T2 for q1
        5e-5, 5e-5,  # T1 and T2 for q2
        5e-5, 5e-5  # T1 and T2 for q3
    ]

    upper_bounds = [
        1, 1, 1,  # Upper bounds for q0
        1, 1, 1,  # Upper bounds for q1
        1, 1, 1,  # Upper bounds for q2
        1, 1, 1,  # Upper bounds for q3
        0.01, 0.01, 0.01,  # ECR error rates upper limits
        0.001, 0.001,  # T1 and T2 for q0
        0.001, 0.001,  # T1 and T2 for q1
        0.001, 0.001,  # T1 and T2 for q2
        0.001, 0.001,  # T1 and T2 for q3
    ]

    bounds = (lower_bounds, upper_bounds)

    options = {'c1': 0.5, 'c2': 0.2, 'w': 0.9}
    optimizer = GlobalBestPSO(n_particles=50, dimensions=len(lower_bounds), options=options, bounds=bounds)

    final_avg_tvd, pos = optimizer.optimize(objective_function, iters=800)

    print("Optimal parameters found:", pos)
    print("Minimum TVD achieved:", final_avg_tvd)
    plt.figure(figsize=(10, 6))
    plt.plot(optimizer.cost_history)
    plt.title('PSO Convergence Curve')
    plt.xlabel('Iterations')
    plt.ylabel('final_avg_tvd')
    plt.show()
