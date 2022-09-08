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
from c3.c3objs import Quantity
from c3.experiment import Experiment
from c3.generator.generator import Generator as Gnr
from c3.model import Model as Mdl
import tensorflow as tf
from c3.optimizers.optimalcontrol import OptimalControl
from c3.parametermap import ParameterMap as PMap
from c3.utils.tf_utils import tf_project_to_comp
from playground.plot_utils import plot_dynamics, plot_signal

np.random.seed(0)


def calc_exp_fid(Experiment: Experiment, index: list) -> float:
    dims = Experiment.pmap.model.dims
    return (1 - fidelities.unitary_infid_set(Experiment.propagators, Experiment.pmap.instructions, index, dims)).numpy()


qubit_lvls = 3
freq_q1 = 5e9
anhar_q1 = -210e6
t1_q1 = 27e-6
t2star_q1 = 39e-6
qubit_temp = 50e-3

q1 = chip.Qubit(name="Q1", desc="Qubit 1",
                freq=Quantity(value=freq_q1, min_val=4.995e9, max_val=5.005e9, unit="Hz 2pi"),
                anhar=Quantity(value=anhar_q1, min_val=-380e6, max_val=-20e6, unit="Hz 2pi"), hilbert_dim=qubit_lvls,
                t1=Quantity(value=t1_q1, min_val=1e-6, max_val=90e-6, unit="s"),
                t2star=Quantity(value=t2star_q1, min_val=10e-6, max_val=90e-3, unit="s"),
                temp=Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"), )

freq_q2 = 5.6e9
anhar_q2 = -240e6
t1_q2 = 23e-6
t2star_q2 = 31e-6
q2 = chip.Qubit(name="Q2", desc="Qubit 2",
                freq=Quantity(value=freq_q2, min_val=5.595e9, max_val=5.605e9, unit='Hz 2pi'),
                anhar=Quantity(value=anhar_q2, min_val=-380e6, max_val=-120e6, unit='Hz 2pi'),
                hilbert_dim=qubit_lvls,
                t1=Quantity(value=t1_q2, min_val=1e-6, max_val=90e-6, unit='s'),
                t2star=Quantity(value=t2star_q2, min_val=10e-6, max_val=90e-6, unit='s'),
                temp=Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit='K')
                )

coupling_strength = 50e6
q1q2 = chip.Coupling(name="Q1-Q2", desc="coupling", comment="Coupling qubit 1 to qubit 2", connected=["Q1", "Q2"],
                     strength=Quantity(value=coupling_strength, min_val=-1 * 1e3, max_val=200e6, unit='Hz 2pi'),
                     hamiltonian_func=hamiltonians.int_XX
                     )

drive = chip.Drive(name="d1", desc="Drive 1", comment="Drive line 1 on qubit 1", connected=["Q1"],
                   hamiltonian_func=hamiltonians.x_drive
                   )
drive2 = chip.Drive(name="d2", desc="Drive 2", comment="Drive line 2 on qubit 2", connected=["Q2"],
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
    name="v_to_hz", V_to_Hz=Quantity(value=v2hz, min_val=0.9e9, max_val=1.1e9, unit="Hz/V")
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
            V_to_Hz=Quantity(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
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

t_final_1Q = 7e-9  # Time for single qubit gates
t_final_2Q = 45e-9  # Time for two qubit gates
tot_gate_time = t_final_2Q + 5 * t_final_1Q
sideband = 50e6

lo_freq_q1 = freq_q1 + sideband
lo_freq_q2 = freq_q2 + sideband

carr_2Q_1 = pulse.Carrier(
    name="carrier",
    desc="Carrier on drive 1",
    params={
        'freq': Quantity(value=lo_freq_q2, min_val=0.9 * lo_freq_q2, max_val=1.1 * lo_freq_q2, unit='Hz 2pi'),
        'framechange': Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit='rad')
    }
)

carr_2Q_2 = pulse.Carrier(
    name="carrier",
    desc="Carrier on drive 2",
    params={
        'freq': Quantity(value=lo_freq_q2, min_val=0.9 * lo_freq_q2, max_val=1.1 * lo_freq_q2, unit='Hz 2pi'),
        'framechange': Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit='rad')
    }
)

comp_gate = gates.Instruction(
    name="comp_gate", targets=[0, 1], t_start=0.0, t_end=tot_gate_time, channels=["d1", "d2"],
    ideal=np.array([
        [0, 0, 0, 1],
        [0, 0, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])
)
gate = comp_gate

gate.add_component(carr_2Q_1, "d1")
gate.add_component(carr_2Q_2, "d2")
gate.comps["d1"]["carrier"].params["framechange"].set_value(
    (-sideband * tot_gate_time) * 2 * np.pi % (2 * np.pi)
)
gate.comps["d2"]["carrier"].params["framechange"].set_value(
    (-sideband * tot_gate_time) * 2 * np.pi % (2 * np.pi)
)

gate_name = gate.get_key()
parameter_map = PMap(instructions=[gate], model=model, generator=generator)
exp = Experiment(pmap=parameter_map)

# rx90 gaussian params
q1_rx90_gauss_params = {
    "amp": Quantity(value=3.236, min_val=0.0, max_val=5, unit="V"),
    "t_final": Quantity(value=t_final_1Q, min_val=0.5 * t_final_1Q, max_val=1.5 * t_final_1Q, unit="s"),
    "xy_angle": Quantity(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Quantity(value=-52.002e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit="Hz 2pi"),
    "delta": Quantity(value=-1, min_val=-5, max_val=3, unit=""),
    "sigma": Quantity(value=t_final_1Q / 4, min_val=t_final_1Q / 8, max_val=t_final_1Q / 2, unit="s"),
}

q1_rx90_gauss_env = pulse.Envelope(
    name="gauss_rx90",
    desc="Gaussian comp for single-qubit gates",
    params=q1_rx90_gauss_params,
    shape=envelopes.gaussian_nonorm,
    normalize_pulse=True
)

# cnot params
q1_cnot_gauss_params = {
    "amp": Quantity(value=123, min_val=0.0, max_val=200, unit="V"),
    "t_final": Quantity(value=tot_gate_time, min_val=0.5 * t_final_2Q, max_val=1.5 * tot_gate_time,
                        unit="s"),
    "xy_angle": Quantity(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Quantity(value=-53.984e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit="Hz 2pi"),
    "delta": Quantity(value=-1, min_val=-5, max_val=3, unit=""),
    "sigma": Quantity(value=t_final_2Q / 4, min_val=t_final_2Q / 8, max_val=t_final_2Q / 2, unit="s"),
}

q1_cnot_gauss_env = pulse.Envelope(
    name="gauss_cnot",
    params=q1_cnot_gauss_params,
    shape=envelopes.gaussian_nonorm,
    normalize_pulse=True
)

q2_cnot_gauss_params = {
    "amp": Quantity(value=2.7, min_val=0.0, max_val=10, unit="V"),
    "t_final": Quantity(value=tot_gate_time, min_val=0.5 * t_final_2Q, max_val=1.5 * tot_gate_time,
                        unit="s"),
    "xy_angle": Quantity(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
    "freq_offset": Quantity(value=-52.994e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit="Hz 2pi"),
    "delta": Quantity(value=-1, min_val=-5, max_val=3, unit=""),
    "sigma": Quantity(value=t_final_2Q / 4, min_val=t_final_2Q / 8, max_val=t_final_2Q / 2, unit="s"),
}

q2_cnot_gauss_env = pulse.Envelope(
    name="gauss_cnot",
    params=q2_cnot_gauss_params,
    shape=envelopes.gaussian_nonorm,
    normalize_pulse=True
)

gate.add_component(q1_rx90_gauss_env, "d1")
gate.add_component(q1_cnot_gauss_env, "d1")
gate.add_component(q2_cnot_gauss_env, "d2")

drivers_signals = gate.comps
drivers_signals = {driver: {sig_name: sig for sig_name, sig in signals.items() if 'carrier' not in sig_name}
                   for driver, signals in drivers_signals.items()}
plot_signal(awg, drivers_signals, tot_gate_time)

exp.set_opt_gates([gate_name])
gate_seq = [gate_name]

psi_init = [[0] * qubit_lvls ** 2]
psi_init[0][1] = 0
psi_init[0][0] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
plot_dynamics(exp, init_state, [gate_name])

print(calc_exp_fid(exp, [0, 1]))
print(abs(tf_project_to_comp(exp.propagators[gate_name], dims=exp.pmap.model.dims).numpy()))

plt.show()
