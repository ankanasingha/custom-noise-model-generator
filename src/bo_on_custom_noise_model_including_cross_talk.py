from qiskit_ibm_runtime import QiskitRuntimeService
from bayes_opt import BayesianOptimization
from qiskit_aer import Aer
from qiskit import QuantumCircuit
from qiskit import transpile
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    depolarizing_error,
    pauli_error,
    thermal_relaxation_error,
    errors
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


def objective_function(selected_qubits, transpiled_circuits, prob_meas1_prep0_q0, prob_meas0_prep1_q0, sx_error_rate_q0,
                       prob_meas1_prep0_q1, prob_meas0_prep1_q1, sx_error_rate_q1,
                       prob_meas1_prep0_q2, prob_meas0_prep1_q2, sx_error_rate_q2,
                       prob_meas1_prep0_q3, prob_meas0_prep1_q3, sx_error_rate_q3,
                       ecr_error_rate_1_0, ecr_error_rate_2_3, ecr_error_rate_2_1, t1_q0, t2_q0, t1_q1, t2_q1, t1_q2,
                       t2_q2, t1_q3, t2_q3, crosstalk_probability, crosstalk_phase_probability):
    qubit_noise_params = {
        0: {'prob_meas1_prep0': prob_meas1_prep0_q0, 'prob_meas0_prep1': prob_meas0_prep1_q0,
            'sx_error_rate': sx_error_rate_q0, 't1': t1_q0, 't2': t2_q0, 'crosstalk_probability': crosstalk_probability,
            'crosstalk_phase_probability': crosstalk_phase_probability},
        1: {'prob_meas1_prep0': prob_meas1_prep0_q1, 'prob_meas0_prep1': prob_meas0_prep1_q1,
            'sx_error_rate': sx_error_rate_q1, 't1': t1_q1, 't2': t2_q1, 'crosstalk_probability': crosstalk_probability,
            'crosstalk_phase_probability': crosstalk_phase_probability},
        2: {'prob_meas1_prep0': prob_meas1_prep0_q2, 'prob_meas0_prep1': prob_meas0_prep1_q2,
            'sx_error_rate': sx_error_rate_q2, 't1': t1_q2, 't2': t2_q2, 'crosstalk_probability': crosstalk_probability,
            'crosstalk_phase_probability': crosstalk_phase_probability},
        3: {'prob_meas1_prep0': prob_meas1_prep0_q3, 'prob_meas0_prep1': prob_meas0_prep1_q3,
            'sx_error_rate': sx_error_rate_q3, 't1': t1_q3, 't2': t2_q3, 'crosstalk_probability': crosstalk_probability,
            'crosstalk_phase_probability': crosstalk_phase_probability}
    }
    ecr_error_rates = {
        '1_0': ecr_error_rate_1_0,
        '2_3': ecr_error_rate_2_3,
        '2_1': ecr_error_rate_2_1
    }

    avg_tvd_by_iteration = []
    initial_noise_model = create_noise_model(ecr_error_rates, qubit_noise_params, selected_qubits)

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
    return -final_avg_tvd


def login_to_ibm_quantum():
    service = QiskitRuntimeService(
        channel='ibm_quantum',
        instance='ibm-q/open/main',
        token=os.environ['IBM_TOKEN']
    )
    return service.backend('ibm_osaka')


def get_bounds():
    return {'prob_meas1_prep0_q0': (1e-8, 1),
            'prob_meas0_prep1_q0': (1e-8, 1),
            'sx_error_rate_q0': (1e-6, 1),
            'prob_meas1_prep0_q1': (1e-8, 1),
            'prob_meas0_prep1_q1': (1e-8, 1),
            'sx_error_rate_q1': (1e-6, 1),
            'prob_meas1_prep0_q2': (1e-8, 1),
            'prob_meas0_prep1_q2': (1e-8, 1),
            'sx_error_rate_q2': (1e-6, 1),
            'prob_meas1_prep0_q3': (1e-8, 1),
            'prob_meas0_prep1_q3': (1e-8, 1),
            'sx_error_rate_q3': (1e-6, 1),
            'ecr_error_rate_1_0': (1e-5, 0.01),
            'ecr_error_rate_2_1': (1e-5, 0.01),
            'ecr_error_rate_2_3': (1e-5, 0.01),
            't1_q0': (5e-5, 0.001),  # T1 and T2 in seconds,
            't2_q0': (5e-5, 0.001),
            't1_q1': (5e-5, 0.001),
            't2_q1': (5e-5, 0.001),
            't1_q2': (5e-5, 0.001),
            't2_q2': (5e-5, 0.001),
            't1_q3': (5e-5, 0.001),
            't2_q3': (5e-5, 0.001),
            'crosstalk_probability': (1e-6, 1e-2),
            'crosstalk_phase_probability': (1e-6, 1e-2)}


def get_initial_points():
    return [
        {'prob_meas1_prep0_q0': 0.023599999,
         'prob_meas0_prep1_q0': 0.0256,
         'sx_error_rate_q0': 0.000134378640,
         'prob_meas1_prep0_q1': 0.008,
         'prob_meas0_prep1_q1': 0.0108,
         'sx_error_rate_q1': 0.000141184005269375,
         'prob_meas1_prep0_q2': 0.0193999999999999,
         'prob_meas0_prep1_q2': 0.0404,
         'sx_error_rate_q2': 0.000291759005495582,
         'prob_meas1_prep0_q3': 0.025,
         'prob_meas0_prep1_q3': 0.0336,
         'sx_error_rate_q3': 0.000424213504448194,
         'ecr_error_rate_1_0': 0.0026819267621233656,
         'ecr_error_rate_2_1': 0.007879566556619033,
         'ecr_error_rate_2_3': 0.010100577411422645,
         'crosstalk_probability': 1e-3,
         'crosstalk_phase_probability': 1e-3,
         't1_q0': 417.416300388374e-6,
         't2_q0': 319.937439214404e-6,
         't1_q1': 229.758984017216e-6,
         't2_q1': 302.385546976999e-6,
         't1_q2': 220.576558661125e-6,
         't2_q2': 263.505016927763e-6,
         't1_q3': 147.724215374594e-6,
         't2_q3': 124.926205308891e-6},
        {'prob_meas1_prep0_q0': 0.0252,
         'prob_meas0_prep1_q0': 0.0749999999999999,
         'sx_error_rate_q0': 0.000113000735139595,
         'prob_meas1_prep0_q1': 0.012,
         'prob_meas0_prep1_q1': 0.0123999999999999,
         'sx_error_rate_q1': 0.000214295537076104,
         'prob_meas1_prep0_q2': 0.029,
         'prob_meas0_prep1_q2': 0.0534,
         'sx_error_rate_q2': 0.00021172705992661,
         'prob_meas1_prep0_q3': 0.0255999999999999,
         'prob_meas0_prep1_q3': 0.0428,
         'sx_error_rate_q3': 0.000258936061790727,
         'ecr_error_rate_1_0': 0.013708445791213136,
         'ecr_error_rate_2_1': 0.006833842370249338,
         'ecr_error_rate_2_3': 0.006221838459653989,
         'crosstalk_probability': 1e-3,
         'crosstalk_phase_probability': 1e-3,
         't1_q0': 289.14421906981e-6,
         't2_q0': 333.030833327208e-6,
         't1_q1': 123.376592714118e-6,
         't2_q1': 208.65250210444e-6,
         't1_q2': 246.1286127637e-6,
         't2_q2': 256.731208569599e-6,
         't1_q3': 358.366032082298e-6,
         't2_q3': 393.9684759492061e-6},
        {'prob_meas1_prep0_q0': 0.0335999999999999,
         'prob_meas0_prep1_q0': 0.0334,
         'sx_error_rate_q0': 0.000122800822400204,
         'prob_meas1_prep0_q1': 0.0066,
         'prob_meas0_prep1_q1': 0.03,
         'sx_error_rate_q1': 0.000168253108443227,
         'prob_meas1_prep0_q2': 0.0232,
         'prob_meas0_prep1_q2': 0.0572,
         'sx_error_rate_q2': 0.000307258052826588,
         'prob_meas1_prep0_q3': 0.0272,
         'prob_meas0_prep1_q3': 0.0356,
         'sx_error_rate_q3': 0.00019476680633718,
         'ecr_error_rate_1_0': 0.0026819267621233656,
         'ecr_error_rate_2_1': 0.007879566556619033,
         'ecr_error_rate_2_3': 0.010100577411422645,
         'crosstalk_probability': 1e-3,
         'crosstalk_phase_probability': 1e-3,
         't1_q0': 256.937193770543e-6,
         't2_q0': 117.468669961055e-6,
         't1_q1': 279.088765247172e-6,
         't2_q1': 326.95684442574e-6,
         't1_q2': 229.135072214554e-6,
         't2_q2': 147.280348215579e-6,
         't1_q3': 371.438478785296e-6,
         't2_q3': 393.968475949206e-6}
    ]


def perform_bo_on_custom_noise_model_including_cross_talk():
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

    bounds = get_bounds()
    initial_points = get_initial_points()
    optimizer = BayesianOptimization(f=objective_function, pbounds=bounds, random_state=2)

    for point in initial_points:
        result = objective_function(selected_qubits, transpiled_circuits, **point)
        optimizer.register(params=point, target=result)

    optimizer.maximize(
        init_points=0,
        n_iter=800
    )

    optimal_params = optimizer.max['params']

    optimal_values = {
        'prob_meas1_prep0': {i: optimal_params[f'prob_meas1_prep0_q{i}'] for i in range(4)},
        'prob_meas0_prep1': {i: optimal_params[f'prob_meas0_prep1_q{i}'] for i in range(4)},
        'sx_error_rate': {i: optimal_params[f'sx_error_rate_q{i}'] for i in range(4)},
        't1': {i: optimal_params[f't1_q{i}'] for i in range(4)},
        't2': {i: optimal_params[f't2_q{i}'] for i in range(4)},
        'crosstalk_probability': optimal_params['crosstalk_probability'],
        'crosstalk_phase_probability': optimal_params['crosstalk_phase_probability'],
        'ecr_error_rate': {
            '1_0': optimal_params['ecr_error_rate_1_0'],
            '2_1': optimal_params['ecr_error_rate_2_1'],
            '2_3': optimal_params['ecr_error_rate_2_3']
        }
    }

    print(f"Optimal measurement error rates for prep0: {optimal_values['prob_meas1_prep0']}")
    print(f"Optimal measurement error rates for prep1: {optimal_values['prob_meas0_prep1']}")
    print(f"Optimal single-qubit gate error rates: {optimal_values['sx_error_rate']}")
    print(f"Optimal T1 relaxation times: {optimal_values['t1']}")
    print(f"Optimal T2 relaxation times: {optimal_values['t2']}")
    print(f"Optimal crosstalk_probability: {optimal_values['crosstalk_probability']}")
    print(f"Optimal crosstalk_phase_probability: {optimal_values['crosstalk_phase_probability']}")
    print(f"Optimal ECR error rates: {optimal_values['ecr_error_rate']}")

    optimal_tvd = -optimizer.max['target']
    print(f"Objective function value (TVD): {optimal_tvd}")

    targets = [-res["target"] for res in optimizer.res]
    plt.figure(figsize=(10, 5))
    plt.plot(targets)
    plt.xlabel('Iteration')
    plt.ylabel('Objective Function Value (TVD)')
    plt.title('Bayesian Optimization Convergence on IBM Osaka')
    plt.show()


if __name__ == "__main__":
    perform_bo_on_custom_noise_model_including_cross_talk()
