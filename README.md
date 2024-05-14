# custom-noise-model-generator

## Introduction
This project creates a custom noise model using different optimisation technique.

## Context

The efficiency of quantum computations is significantly obstructed by quantum noise. This leads to errors in quantum gate operations, state preparation, and measurements. In our study we are using IBM Quantum computer, particularly, IBM Osaka. The hypothesis is that the noise model provided by IBM quantum for their Quantum Computer hardware is not accurate, and therefore needs a custom noise model to closely simulate the noise characteristics observed in real quantum environments. Hence, this thesis first investigates the discrepancies between IBM’s provided quantum noise models and the actual noise encountered on IBM quantum hardware. Using a series of randomly generated quantum circuits, this research conducts a comparative study by running these circuits on real IBM quantum computers and simulating them using IBM's noise model. The Total Variation Distance (TVD) metric is employed to quantitatively assess the similarity between the experimental counts and the simulated counts. The findings reveal significant discrepancies in the IBM noise models for 4-qubit systems, while 2-qubit systems show relatively minor discrepancies. These discrepancies underline the limitations of IBM's current noise models, particularly for more complex quantum systems.

To address these limitations, a custom noise model is proposed, modeling gate errors, crosstalk, SPAM errors, and relaxation times. The noise model's parameters were optimized using both Bayesian Optimization and Particle Swarm Optimization. Bayesian Optimization has shown a better performance in minimizing the TVD between the real and simulated data while Particle Swarm Optimization gave realistic noise parameters. Data also shows that the inclusion of crosstalk in the noise model has improved its accuracy in terms of a lower TVD. It has also shown that the optimized custom noise model is better than the IBM-provided models, as the earlier one has a lower TVD value.

This thesis presents a methodology that has been applied to 2q and 4q configurations in IBM Osaka, however, it could be applied to other superconducting IBM quantum hardwares as well.
The custom noise model can be used to generate simulated datasets that reflect real-world conditions for training and testing quantum machine learning algorithms. This is useful in scenarios where experimental data is limited or expensive to obtain. This study reveals the viability of developing more accurate and flexible noise models using sophisticated optimization approaches, to prepare for more reliable quantum computing.

##  Development

### Prerequisites
- python 3.10 and above
- pip
- API Token from IBM Quantum Platform. This token needs to be injected as environment variable named  `IBM_TOKEN` during execution of the subsequent programs.

### Running the application
