from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_aer import Aer
from qiskit import QuantumCircuit, transpile
import numpy as np
import os
from qiskit_aer.noise import (
    NoiseModel,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error
)


def create_noise_model(ecr_error_rates, qubit_noise_params):
    noise_model = NoiseModel()
    selected_qubits = [0, 1, 2, 3]
    for i in selected_qubits:
        # Preparation errors
        prep_errors = pauli_error(
            [('X', qubit_noise_params[i]['prob_meas1_prep0']), ('I', 1 - qubit_noise_params[i]['prob_meas1_prep0'])])
        noise_model.add_quantum_error(prep_errors, 'reset', [i])

    for qubit, params in qubit_noise_params.items():
        sx_error = pauli_error([('X', params['sx_error_rate']), ('I', 1 - params['sx_error_rate'])])
        noise_model.add_quantum_error(sx_error, 'sx', [qubit])

        # Adding correlated SX dephasing errors to neighboring qubits
        for other_qubit in qubit_noise_params:
            if other_qubit != qubit and abs(other_qubit - qubit) == 1:  # Check if qubits are neighbors
                crosstalk_phase_error = pauli_error(
                    [('Z', params['crosstalk_phase_probability']), ('I', 1 - params['crosstalk_phase_probability'])])
                noise_model.add_quantum_error(crosstalk_phase_error, 'sx', [other_qubit])

    for pair, error_rate in ecr_error_rates.items():
        qubits = [int(q) for q in pair.split('_')]
        control, target = qubits[0], qubits[1]

        # Define two-qubit depolarizing error for the ecr gate
        ecr_error = depolarizing_error(error_rate, 2)
        noise_model.add_quantum_error(ecr_error, 'ecr', qubits)

        prob_control = qubit_noise_params[control]['crosstalk_probability']
        prob_target = qubit_noise_params[target]['crosstalk_probability']

        # Calculate probabilities for 'IZ' and 'ZI'
        prob_IZ = prob_control * (1 - prob_target)
        prob_ZI = prob_target * (1 - prob_control)

        # Calculate 'II' probability to ensure total is exactly 1
        prob_II = 1 - prob_IZ - prob_ZI

        # Define crosstalk errors with corrected probabilities
        crosstalk_error = pauli_error([
            ('IZ', prob_IZ),
            ('ZI', prob_ZI),
            ('II', prob_II)
        ])
        noise_model.add_quantum_error(crosstalk_error, 'ecr', qubits)

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


def calculate_tvd_for_ibm_noise_model(transpiled_qc, backend, real_counts):
    ibm_noise_model = NoiseModel.from_backend(backend)
    simulator = Aer.get_backend('qasm_simulator')
    ibm_simulated_job = simulator.run(transpiled_qc, noise_model=ibm_noise_model, shots=8192, seed_simulator=42,
                                      optimization_level=0)
    ibm_simulated_results = ibm_simulated_job.result()
    ibm_simulated_counts = ibm_simulated_results.get_counts()
    print("IBM simulated Counts:", ibm_simulated_counts)
    ibm_tvd_value = calculate_tvd(ibm_simulated_counts, real_counts)
    # Print TVD values using IBM's Noise Model||
    print("Calculated TVD for IBM's Noise Model:", ibm_tvd_value)


def create_test_circuit():
    qc = QuantumCircuit(4)
    # Apply ECR gates predominantly
    ecr_cycles = 4  # Number of cycles of ECR gates through the chosen pairs to reach approximately 16
    for _ in range(ecr_cycles):
        # ECR gates between chosen pairs for structured connectivity
        qc.ecr(0, 1)
        qc.ecr(1, 2)
        qc.ecr(2, 3)
        qc.ecr(3, 0)
    # Add SX and RZ gates to use up the remaining depth
    # Distribute 4 SX and 4 RZ gates evenly
    for i in range(4):
        qc.sx(i)  # Apply SX gate on each qubit
        qc.rz(np.pi / 4, i)  # Apply RZ gate with pi/4 phase on each qubit
    qc.measure_all()
    # Draw the circuit
    print(qc.draw())
    return qc


def calculate_tvd_for_pso_optimised(transpiled_qc, real_counts):
    qubit_noise_params_PSO = {
        0: {'prob_meas1_prep0': 4.70668720e-01, 'prob_meas0_prep1': 1.71453110e-01, 'sx_error_rate': 5.25649013e-01,
            't1': 8.12808658e-04, 't2': 6.18312267e-04, 'crosstalk_probability': 3.14766548e-03,
            'crosstalk_phase_probability': 5.80475160e-03},
        1: {'prob_meas1_prep0': 2.69883870e-01, 'prob_meas0_prep1': 6.78626284e-01, 'sx_error_rate': 6.63957445e-01,
            't1': 5.26142826e-04, 't2': 4.41938363e-04, 'crosstalk_probability': 3.14766548e-03,
            'crosstalk_phase_probability': 5.80475160e-03},
        2: {'prob_meas1_prep0': 4.18224199e-05, 'prob_meas0_prep1': 6.18366447e-01, 'sx_error_rate': 7.05366376e-01,
            't1': 5.99895724e-04, 't2': 5.72684517e-04, 'crosstalk_probability': 3.14766548e-03,
            'crosstalk_phase_probability': 5.80475160e-03},
        3: {'prob_meas1_prep0': 4.55503679e-01, 'prob_meas0_prep1': 2.75956555e-01, 'sx_error_rate': 4.50746270e-01,
            't1': 8.58618154e-04, 't2': 9.31968624e-04, 'crosstalk_probability': 3.14766548e-03,
            'crosstalk_phase_probability': 5.80475160e-03}
    }
    ecr_error_rates_PSO = {
        '1_0': 4.11903488e-03,
        '2_3': 5.33120660e-03,
        '2_1': 6.06829010e-03
    }
    selected_qubits = [0, 1, 2, 3]
    optimized_noise_model_PSO = create_noise_model(ecr_error_rates_PSO, qubit_noise_params_PSO)
    simulator = Aer.get_backend('qasm_simulator')
    optimized_simulated_job = simulator.run(transpiled_qc, noise_model=optimized_noise_model_PSO, shots=8192,
                                            seed_simulator=42, optimization_level=0)
    optimized_simulated_results = optimized_simulated_job.result()
    optimized_simulated_counts = optimized_simulated_results.get_counts()
    print("simulated Counts:", optimized_simulated_counts)
    optimized_tvd_value = calculate_tvd(optimized_simulated_counts, real_counts)
    print("Calculated TVD after PSO Optimization:", optimized_tvd_value)


def calculate_tvd_for_bo_optimized(transpiled_qc, real_counts):
    qubit_noise_params_BO = {
        0: {'prob_meas1_prep0': 1e-08, 'prob_meas0_prep1': 1e-08, 'sx_error_rate': 1e-06, 't1': 5e-05, 't2': 5e-05,
            'crosstalk_probability': 1e-06, 'crosstalk_phase_probability': 0.01},
        1: {'prob_meas1_prep0': 1e-08, 'prob_meas0_prep1': 1e-08, 'sx_error_rate': 1e-06, 't1': 5e-05, 't2': 5e-05,
            'crosstalk_probability': 1e-06, 'crosstalk_phase_probability': 0.01},
        2: {'prob_meas1_prep0': 1e-08, 'prob_meas0_prep1': 1e-08, 'sx_error_rate': 1.0, 't1': 5e-05, 't2': 5e-05,
            'crosstalk_probability': 1e-06, 'crosstalk_phase_probability': 0.01},
        3: {'prob_meas1_prep0': 1e-08, 'prob_meas0_prep1': 1e-08, 'sx_error_rate': 1e-06, 't1': 0.001, 't2': 0.001,
            'crosstalk_probability': 1e-06, 'crosstalk_phase_probability': 0.01}
    }
    ecr_error_rates_BO = {
        '1_0': 0.01,
        '2_3': 1e-05,
        '2_1': 1e-05
    }
    optimized_noise_model_BO = create_noise_model(ecr_error_rates_BO, qubit_noise_params_BO)
    simulator = Aer.get_backend('qasm_simulator')
    optimized_simulated_job_BO = simulator.run(transpiled_qc, noise_model=optimized_noise_model_BO, shots=8192,
                                               seed_simulator=42, optimization_level=0)
    optimized_simulated_results_BO = optimized_simulated_job_BO.result()
    optimized_simulated_counts_BO = optimized_simulated_results_BO.get_counts()
    print("simulated Counts:", optimized_simulated_counts_BO)
    optimized_tvd_value_BO = calculate_tvd(optimized_simulated_counts_BO, real_counts)
    print("Calculated TVD after BO Optimization:", optimized_tvd_value_BO)


def create_test_circuit_and_validate_model():
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
        token=os.environ['IBM_TOKEN']
    )
    backend = service.backend('ibm_osaka')
    qc = create_test_circuit()

    transpiled_qc = transpile(qc, backend)
    # Execute the transpiled circuits on the real quantum computer
    real_job_id = 'crx7nz77wv80008fkzxg'
    real_job = service.job(real_job_id)

    # Collect the results
    real_results = real_job.result()
    real_counts = real_results.get_counts()
    print("Real Counts:", real_counts)

    calculate_tvd_for_pso_optimised(transpiled_qc, real_counts)

    calculate_tvd_for_ibm_noise_model(transpiled_qc, backend, real_counts)

    calculate_tvd_for_bo_optimized(transpiled_qc, real_counts)


if __name__ == "__main__":
    create_test_circuit_and_validate_model()

