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
import tensorflow as tf
from c3.optimizers.optimalcontrol import OptimalControl
from c3.parametermap import ParameterMap as PMap
from c3.utils.tf_utils import tf_project_to_comp

np.random.seed(0)

qubit_lvls = 3
freq_q1 = 5e9
anhar_q1 = -210e6
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

freq_q2 = 5.6e9
anhar_q2 = -240e6
t1_q2 = 23e-6
t2star_q2 = 31e-6
q2 = chip.Qubit(
    name="Q2",
    desc="Qubit 2",
    freq=Qty(
        value=freq_q2,
        min_val=5.595e9,
        max_val=5.605e9,
        unit='Hz 2pi'
    ),
    anhar=Qty(
        value=anhar_q2,
        min_val=-380e6,
        max_val=-120e6,
        unit='Hz 2pi'
    ),
    hilbert_dim=qubit_lvls,
    t1=Qty(
        value=t1_q2,
        min_val=1e-6,
        max_val=90e-6,
        unit='s'
    ),
    t2star=Qty(
        value=t2star_q2,
        min_val=10e-6,
        max_val=90e-6,
        unit='s'
    ),
    temp=Qty(
        value=qubit_temp,
        min_val=0.0,
        max_val=0.12,
        unit='K'
    )
)

coupling_strength = 50e6
q1q2 = chip.Coupling(
    name="Q1-Q2",
    desc="coupling",
    comment="Coupling qubit 1 to qubit 2",
    connected=["Q1", "Q2"],
    strength=Qty(
        value=coupling_strength,
        min_val=-1 * 1e3,
        max_val=200e6,
        unit='Hz 2pi'
    ),
    hamiltonian_func=hamiltonians.int_XX
)

drive = chip.Drive(
    name="d1",
    desc="Drive 1",
    comment="Drive line 1 on qubit 1",
    connected=["Q1"],
    hamiltonian_func=hamiltonians.x_drive
)
drive2 = chip.Drive(
    name="d2",
    desc="Drive 2",
    comment="Drive line 2 on qubit 2",
    connected=["Q2"],
    hamiltonian_func=hamiltonians.x_drive
)

model = Mdl(
    [q1, q2],  # Individual, self-contained components
    [drive, drive2, q1q2],  # Interactions between components
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

# t_final = 7e-9  # Time for single qubit gates
t_final = 45e-9  # Time for two qubit gates
sideband = 50e6

# gaussiam params
gauss_params = {
    "amp": Qty(value=0.36, min_val=0.0, max_val=5, unit="V"),
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

gauss2_params = copy.deepcopy(gauss_params)
gauss2_params['amp'].set_value(0)
gauss2_env = copy.deepcopy(gauss_env)
gauss2_env.params = gauss2_params

lo_freq_q1 = freq_q1 + sideband
lo_freq_q2 = freq_q2 + sideband

carr_2Q_1 = pulse.Carrier(
    name="carrier",
    desc="Carrier on drive 1",
    params={
        'freq': Qty(value=lo_freq_q2, min_val=0.9 * lo_freq_q2, max_val=1.1 * lo_freq_q2, unit='Hz 2pi'),
        'framechange': Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit='rad')
    }
)

carr_2Q_2 = pulse.Carrier(
    name="carrier",
    desc="Carrier on drive 2",
    params={
        'freq': Qty(value=lo_freq_q2, min_val=0.9 * lo_freq_q2, max_val=1.1 * lo_freq_q2, unit='Hz 2pi'),
        'framechange': Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit='rad')
    }
)

# lo_freq_q1 = 5e9 + sideband
# carrier_parameters = {
#     "freq": Qty(value=lo_freq_q1, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
#     "framechange": Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
# }
# carr = pulse.Carrier(
#     name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
# )
#
# rx90p_q1 = gates.Instruction(
#     name="rx90p", targets=[0], t_start=0.0, t_end=t_final, channels=["d1"]
# )
#
# rx90p_q1.add_component(carr, "d1")
# rx90p_q1.add_component(gauss_env, "d1")
#
# single_q_gates = [rx90p_q1]
# CNOT comtrolled by qubit 1
cnot12 = gates.Instruction(
    name="cx", targets=[0, 1], t_start=0.0, t_end=t_final, channels=["d1", "d2"],
    ideal=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0]
    ])
)

cnot12.add_component(carr_2Q_1, "d1")
cnot12.add_component(carr_2Q_2, "d2")
cnot12.comps["d1"]["carrier"].params["framechange"].set_value(
    (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
)

cnot12.add_component(gauss_env, 'd1')
cnot12.add_component(gauss2_env, 'd2')

gate_name = cnot12.name + '[0, 1]'

parameter_map = PMap(instructions=[cnot12], model=model, generator=generator)
exp = Exp(pmap=parameter_map)
exp.set_opt_gates([gate_name])
unitaries = exp.compute_propagators()
gate_seq = [gate_name]

opt_gates = [gate_name]


def get_fid_over_time():
    freqs = {}
    framechanges = {}
    for line, ctrls in cnot12.comps.items():
        offset = tf.constant(0.0, tf.float64)
        for ctrl in ctrls.values():
            if "freq_offset" in ctrl.params.keys():
                if ctrl.params["amp"] != 0.0:
                    offset = ctrl.params["freq_offset"].get_value()
        freqs[line] = tf.cast(
            ctrls["carrier"].params["freq"].get_value() + offset,
            tf.complex128,
        )
        framechanges[line] = tf.cast(
            ctrls["carrier"].params["framechange"].get_value(),
            tf.complex128,
        )

    times = exp.ts.numpy()
    partial_props = exp.partial_propagators[gate_name].numpy()
    ideal_gate = cnot12.ideal
    n = ideal_gate.shape[0]
    U = np.eye(9)
    fid_over_t = []
    cost = 0
    dt = times[1] - times[0]
    for dU, t in zip(partial_props, times):
        U = dU @ U
        # rot_mat = base_rot_mat @ rot_mat

        w1 = 2 * np.pi * 4.995380e9
        w2 = 2 * np.pi * 5.6036557e9
        w_sum = 2 * np.pi * 1.05919032e+10
        passive_rot = 2 * np.pi * -2.36028622e+05

        rot_t = t
        # rotated_U = U * np.array([[np.exp(1j * passive_rot * rot_t)], [np.exp(1j * w2 * rot_t)],
        #                           [np.exp(1j * w1 * rot_t)], [np.exp(1j * w_sum * rot_t)]])

        rot_mat = model.get_Frame_Rotation(rot_t, freqs, framechanges).numpy()
        rotated_U = tf_project_to_comp(rot_mat @ U, dims=[3, 3])
        print(U[1, 3], t)

        fid = abs(np.trace(scipy.linalg.fractional_matrix_power(ideal_gate.conj().T, t / t_final) @ rotated_U) / n) ** 2

        cost += fid * (1 - np.exp(-t / t_final))

        fid_over_t.append({'t': t, 'F': fid})

    print(cost * dt / t_final * np.e)
    print(1 - fidelities.unitary_infid(cnot12.ideal.astype(np.complex128), exp.propagators['cx[0, 1]'],
                                       dims=[3, 3]).numpy()[0])
    print(fid_over_t[-1]['F'])
    print()

    return pd.DataFrame.from_records(fid_over_t)


for v in np.linspace(1, 3, 3):
    gauss_env.params['amp'].set_value(v)

    print(v)
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
