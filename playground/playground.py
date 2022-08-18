# System imports
import copy
import os
import tempfile

import numpy as np

# Building blocks
import pandas as pd
import scipy.linalg
from matplotlib import pyplot as plt

import c3.generator.devices as devices

# Libs and helpers
import c3.libraries.algorithms as algorithms
import c3.libraries.chip as chip
import c3.libraries.envelopes as envelopes
import c3.libraries.fidelities as fidelities
import c3.libraries.hamiltonians as hamiltonians
import c3.libraries.tasks as tasks
import c3.signal.gates as gates
import c3.signal.pulse as pulse

# Main C3 objects
from c3.c3objs import Quantity as Qty
from c3.experiment import Experiment as Exp
from c3.generator.generator import Generator as Gnr
from c3.model import Model as Mdl
from c3.optimizers.optimalcontrol import OptimalControl
from c3.parametermap import ParameterMap as PMap

np.random.seed(0)

# Qiskit related modules

qubit_lvls = 3
freq_q1 = 5e9
anhar_q1 = -100e6
t1_q1 = 27e-6
t2star_q1 = 39e-6
qubit_temp = 50e-3

q1 = chip.Qubit(
    name="Q1",
    desc="Qubit 1",
    freq=Qty(value=freq_q1, min_val=4.995e9, max_val=5.005e9, unit="Hz 2pi"),
    anhar=Qty(value=anhar_q1, min_val=-380e6, max_val=-20e6, unit="Hz 2pi"),
    hilbert_dim=qubit_lvls,
    t1=Qty(value=t1_q1, min_val=1e-6, max_val=90e-6, unit="s"),
    t2star=Qty(value=t2star_q1, min_val=10e-6, max_val=90e-3, unit="s"),
    temp=Qty(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
)

drive = chip.Drive(
    name="d1",
    desc="Drive 1",
    comment="Drive line 1 on qubit 1",
    connected=["Q1"],
    hamiltonian_func=hamiltonians.x_drive,
)

init_temp = 50e-3
init_ground = tasks.InitialiseGround(
    init_temp=Qty(value=init_temp, min_val=-0.001, max_val=0.22, unit="K")
)

model = Mdl(
    [q1],  # Individual, self-contained components
    [drive],  # Interactions between components
)

model.set_lindbladian(False)
model.set_dressed(True)

sim_res = 100e9  # Resolution for numerical simulation
awg_res = 2e9  # Realistic, limited resolution of an AWG
lo = devices.LO(name="lo", resolution=sim_res)
awg = devices.AWG(name="awg", resolution=awg_res)
mixer = devices.Mixer(name="mixer")

dig_to_an = devices.DigitalToAnalog(name="dac", resolution=sim_res)
v2hz = 1e9
v_to_hz = devices.VoltsToHertz(
    name="v_to_hz", V_to_Hz=Qty(value=v2hz, min_val=0.9e9, max_val=1.1e9, unit="Hz/V")
)

generator = Gnr(
    devices={
        "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
        "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
        "DigitalToAnalog": devices.DigitalToAnalog(
            name="dac", resolution=sim_res, inputs=1, outputs=1
        ),
        "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
        "VoltsToHertz": devices.VoltsToHertz(
            name="v_to_hz",
            V_to_Hz=Qty(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
            inputs=1,
            outputs=1,
        ),
    },
    chains={
        "d1": {
            "LO": [],
            "AWG": [],
            "DigitalToAnalog": ["AWG"],
            "Mixer": ["LO", "DigitalToAnalog"],
            "VoltsToHertz": ["Mixer"],
        },
        "d2": {
            "LO": [],
            "AWG": [],
            "DigitalToAnalog": ["AWG"],
            "Mixer": ["LO", "DigitalToAnalog"],
            "VoltsToHertz": ["Mixer"],
        },
    },
)

t_final = 7e-9  # Time for single qubit gates
sideband = 50e6

# gaussiam params
gauss_params = {
    "amp": Qty(value=0.36, min_val=0.0, max_val=2.5, unit="V"),
    "t_final": Qty(
        value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
    ),
    "xy_angle": Qty(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Qty(
        value=-sideband - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit="Hz 2pi"
    ),
    "delta": Qty(value=-1, min_val=-5, max_val=3, unit=""),
    "sigma": Qty(value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"),
}

gauss_env = pulse.Envelope(
    name="gauss",
    desc="Gaussian comp for single-qubit gates",
    params=gauss_params,
    shape=envelopes.gaussian_nonorm,
)

lo_freq_q1 = 5e9 + sideband
carrier_parameters = {
    "freq": Qty(value=lo_freq_q1, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
    "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
}
carr = pulse.Carrier(
    name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
)

rx90p_q1 = gates.Instruction(
    name="rx90p", targets=[0], t_start=0.0, t_end=t_final, channels=["d1"]
)

rx90p_q1.add_component(carr, "d1")
rx90p_q1.add_component(gauss_env, "d1")

single_q_gates = [rx90p_q1]

parameter_map = PMap(instructions=single_q_gates, model=model, generator=generator)
exp = Exp(pmap=parameter_map)
exp.set_opt_gates(["rx90p[0]"])
unitaries = exp.compute_propagators()
gate_seq = ["rx90p[0]"]

opt_gates = ["rx90p[0]"]


def get_fid_over_time():
    partial_props = exp.partial_propagators['rx90p[0]'].numpy()
    ideal_gate = rx90p_q1.ideal
    n = ideal_gate.shape[0]
    U = np.eye(n)
    fid_over_t = []
    cost = 0
    times = exp.ts.numpy()
    for dU, t in zip(partial_props, times):
        U = dU[:2, :2] @ U

        rotated_U = U * np.array([[1], [np.exp(1j * 2 * np.pi * freq_q1 * t)]])
        fid = abs(np.trace(scipy.linalg.fractional_matrix_power(ideal_gate.conj().T, t / t_final) @ rotated_U) / n) ** 2

        cost += fid * (1 - np.exp(-t / t_final))

        fid_over_t.append({'t': t, 'F': fid})

    print(cost * (times[1] - times[0]) / t_final * np.e)

    return pd.DataFrame.from_records(fid_over_t)


for v in np.linspace(0, 2.5, 30):
    gauss_env.params['amp'].set_value(v)

    exp.compute_propagators()

    fid_over_t = get_fid_over_time()
    fid_over_t.plot(x='t', y='F', label=f'{v:.2f}', ax=plt.gca())
    plt.annotate(f'{v:.2f}', (fid_over_t.t.iat[-1], fid_over_t.F.iat[-1]))

plt.show()

# gateset_opt_map = [
#     [
#         ("rx90p[0]", "d1", "carrier", "framechange"),
#     ],
#     [
#         ("rx90p[0]", "d1", "gauss", "amp"),
#     ],
#     [
#         ("rx90p[0]", "d1", "gauss", "freq_offset"),
#     ],
# ]
#
# parameter_map.set_opt_map(gateset_opt_map)
#
# # Create a temporary directory to store logfiles, modify as needed
# log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")
#
# opt = OptimalControl(
#     dir_path=log_dir,
#     fid_func=fidelities.sparse_unitary_infid_set,
#     fid_subspace=["Q1"],
#     pmap=exp.pmap,
#     algorithm=algorithms.lbfgs,
#     options={'maxfun': 50},
#     run_name="temp",
# )
#
# exp.set_opt_gates(opt_gates)
# opt.set_exp(exp)
# opt_res = opt.optimize_controls()
#
# parameter_map.print_parameters()
