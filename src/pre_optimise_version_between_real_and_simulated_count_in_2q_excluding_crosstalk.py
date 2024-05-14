from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit import transpile
import json
import os
import numpy as np
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error
)

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


def create_noise_model(ecr_error_rates, qubit_noise_params, selected_qubits):
    noise_model = NoiseModel()

    for i in selected_qubits:
    # Preparation errors
        prep_errors = pauli_error([('X', qubit_noise_params[i]['prob_meas1_prep0']), ('I', 1 -  qubit_noise_params[i]['prob_meas1_prep0'])])
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

            thermal_error = thermal_relaxation_error(qubit_noise_params[qubit]['t1'], qubit_noise_params[qubit]['t2'], 660e-9)  # using 660ns as gate time
            noise_model.add_quantum_error(thermal_error, 'measure', [qubit])  # Apply to measurement

    for i in selected_qubits:
        readout_error = ReadoutError([[1 - qubit_noise_params[i]['prob_meas1_prep0'], qubit_noise_params[i]['prob_meas1_prep0']],
                                      [qubit_noise_params[i]['prob_meas0_prep1'], 1 - qubit_noise_params[i]['prob_meas0_prep1']]])
        noise_model.add_readout_error(readout_error, [i])

    return noise_model


def calculate_tvd(simulated_counts, real_counts):
    sim_probs = {k: v / 8192 for k, v in simulated_counts.items()}
    real_probs = {k: v / 8192 for k, v in real_counts.items()}
    return 0.5 * sum(abs(sim_probs.get(k, 0) - real_probs.get(k, 0)) for k in set(sim_probs) | set(real_probs))


def login_to_ibm_quantum():
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
        token=os.environ['IBM_TOKEN']
    )
    return service.backend('ibm_osaka')


def pre_optimise_version_between_real_and_simulated_count_in_2q_excluding_crosstalk():
    backend = login_to_ibm_quantum()
    selected_qubits = [0, 1, 2, 3]
    num_circuits = 100
    depth = 20
    ecr_probability = 0.8

    # Generate the random circuits
    random_circuits = [create_random_circuits(selected_qubits, depth, ecr_probability) for _ in range(num_circuits)]

    # Transpile the circuits for the backend
    transpiled_circuits = transpile(random_circuits,
                                    backend=backend,
                                    initial_layout=selected_qubits,
                                    optimization_level=0)

    prob_meas1_prep0_q0 = 0.0236
    prob_meas0_prep1_q0 = 0.0256
    sx_error_rate_q0 = 0.000134378640
    prob_meas1_prep0_q1 = 0.008
    prob_meas0_prep1_q1 = 0.0108
    sx_error_rate_q1 = 0.000141184005269375
    prob_meas1_prep0_q2 = 0.0194
    prob_meas0_prep1_q2 = 0.0404
    sx_error_rate_q2 = 0.000291759005495582
    prob_meas1_prep0_q3 = 0.025
    prob_meas0_prep1_q3 = 0.0336
    sx_error_rate_q3 = 0.000424213504448194
    ecr_error_rate_1_0 = 0.0026819267621233656
    ecr_error_rate_2_1 = 0.007879566556619033
    ecr_error_rate_2_3 = 0.010100577411422645
    t1_q0 = 417.416300388374e-6
    t2_q0 = 319.937439214404e-6
    t1_q1 = 229.758984017216e-6
    t2_q1 = 302.385546976999e-6
    t1_q2 = 220.576558661125e-6
    t2_q2 = 263.505016927763e-6
    t1_q3 = 147.724215374594e-6
    t2_q3 = 124.926205308891e-6
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
    initial_noise_model = create_noise_model(ecr_error_rates, qubit_noise_params, selected_qubits)

    for i in range(1, 6):
        simulator = Aer.get_backend('qasm_simulator')
        simulated_job = simulator.run(transpiled_circuits, noise_model=initial_noise_model, shots=8192,
                                      seed_simulator=42, optimization_level=0)

        simulated_results = simulated_job.result()
        simulated_counts = [simulated_results.get_counts(j) for j in range(100)]

        real_counts_file_path = os.path.join(os.path.dirname(os.getcwd()), "resources", "2q", "real_counts",
                                             f"real_counts_{i}.json")
        with open(real_counts_file_path, 'r') as file:
            real_counts = json.load(file)
        avg_tvd = np.mean([calculate_tvd(simulated_counts[i], real_counts[i]) for i in range(len(real_counts))])
        avg_tvd_by_iteration.append(avg_tvd)

    final_avg_tvd = sum(avg_tvd_by_iteration) / 5
    print(final_avg_tvd)


if __name__ == "__main__":
    pre_optimise_version_between_real_and_simulated_count_in_2q_excluding_crosstalk()
