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

# freq_q2 = 5.6e9
# anhar_q2 = -240e6
# t1_q2 = 23e-6
# t2star_q2 = 31e-6
# q2 = chip.Qubit(
#     name="Q2",
#     desc="Qubit 2",
#     freq=Qty(
#         value=freq_q2,
#         min_val=5.595e9,
#         max_val=5.605e9,
#         unit='Hz 2pi'
#     ),
#     anhar=Qty(
#         value=anhar_q2,
#         min_val=-380e6,
#         max_val=-120e6,
#         unit='Hz 2pi'
#     ),
#     hilbert_dim=qubit_lvls,
#     t1=Qty(
#         value=t1_q2,
#         min_val=1e-6,
#         max_val=90e-6,
#         unit='s'
#     ),
#     t2star=Qty(
#         value=t2star_q2,
#         min_val=10e-6,
#         max_val=90e-6,
#         unit='s'
#     ),
#     temp=Qty(
#         value=qubit_temp,
#         min_val=0.0,
#         max_val=0.12,
#         unit='K'
#     )
# )
#
# coupling_strength = 50e6
# q1q2 = chip.Coupling(
#     name="Q1-Q2",
#     desc="coupling",
#     comment="Coupling qubit 1 to qubit 2",
#     connected=["Q1", "Q2"],
#     strength=Qty(
#         value=coupling_strength,
#         min_val=-1 * 1e3,
#         max_val=200e6,
#         unit='Hz 2pi'
#     ),
#     hamiltonian_func=hamiltonians.int_XX
# )

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
    [q1],  # Individual, self-contained components
    [drive],  # Interactions between components
)
# model = Mdl(
#     [q1, q2],  # Individual, self-contained components
#     [drive, drive2, q1q2],  # Interactions between components
# )

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
# t_final = 45e-9  # Time for two qubit gates
sideband = 50e6

# gaussian params
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

# gauss2_params = copy.deepcopy(gauss_params)
# gauss2_params['amp'].set_value(0)
# gauss2_env = copy.deepcopy(gauss_env)
# gauss2_env.params = gauss2_params
#
# lo_freq_q1 = freq_q1 + sideband
# lo_freq_q2 = freq_q2 + sideband
#
# carr_2Q_1 = pulse.Carrier(
#     name="carrier",
#     desc="Carrier on drive 1",
#     params={
#         'freq': Qty(value=lo_freq_q2, min_val=0.9 * lo_freq_q2, max_val=1.1 * lo_freq_q2, unit='Hz 2pi'),
#         'framechange': Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit='rad')
#     }
# )
#
# carr_2Q_2 = pulse.Carrier(
#     name="carrier",
#     desc="Carrier on drive 2",
#     params={
#         'freq': Qty(value=lo_freq_q2, min_val=0.9 * lo_freq_q2, max_val=1.1 * lo_freq_q2, unit='Hz 2pi'),
#         'framechange': Qty(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit='rad')
#     }
# )

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
gate = rx90p_q1
gate_name = gate.name + '[0]'

# cnot12 = gates.Instruction(
#     name="cx", targets=[0, 1], t_start=0.0, t_end=t_final, channels=["d1", "d2"],
#     ideal=np.array([
#         [1, 0, 0, 0],
#         [0, 1, 0, 0],
#         [0, 0, 0, 1],
#         [0, 0, 1, 0]
#     ])
# )
#
# cnot12.add_component(carr_2Q_1, "d1")
# cnot12.add_component(carr_2Q_2, "d2")
# cnot12.comps["d1"]["carrier"].params["framechange"].set_value(
#     (-sideband * t_final) * 2 * np.pi % (2 * np.pi)
# )
#
# cnot12.add_component(gauss_env, 'd1')
# cnot12.add_component(gauss2_env, 'd2')
#
# gate = cnot12
# gate_name = gate.name + '[0, 1]'

parameter_map = PMap(instructions=[gate], model=model, generator=generator)
exp = Exp(pmap=parameter_map)


def get_fid_func(mu: float, F_bar: float):
    def fid_func(*args, **kwargs):
        avg_fid = 1 - fidelities.unitary_infid_set(*args, **kwargs)

        reg_term = fidelities.amplitude_regularization_cost(*args, loss_func_type='sqrt', **kwargs)

        return reg_term - mu * tf.math.log(avg_fid - F_bar)

    return fid_func


def run_log_barrier_opt_iter(exp_opt: OptimalControl, mu: float, F_bar: float):
    assert 0 < F_bar <= 1, 'Error: F_bar is OOB!'

    fid_func = get_fid_func(mu, F_bar)

    exp_opt.set_fid_func(fid_func)

    exp_opt.optimize_controls()


def set_amps_to_zero(exp: Exp):
    for instruction in exp.pmap.instructions.values():
        for channel in instruction.comps.values():
            for component in channel.values():
                if "amp" in component.params:
                    component.params['amp'].set_value(0)


def setup_experiment_opt_ctrl(exp: Exp, maxiter: int = 50) -> OptimalControl:
    log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

    opt = OptimalControl(
        dir_path=log_dir,
        fid_func=None,
        fid_subspace=["Q1"],  # TODO-set this automatically
        pmap=exp.pmap,
        algorithm=algorithms.lbfgs,
        options={'maxiter': maxiter},
    )

    opt.set_exp(exp)

    return opt


def calc_exp_fid(exp: Exp, index: list, dims: list) -> float:
    return 1 - fidelities.unitary_infid_set(exp.propagators, exp.pmap.instructions, index, dims)


def signal_opt_via_log_barrier(exp: Exp, opt_q_inds: list, opt_q_dims: list, alpha: float = 2,
                               init_F_bar_mul: float = 0.9, init_mu: float = 1e-6, verbose: bool = True):
    assert 1 < alpha, 'Error: alpha is OOB!'
    assert exp.opt_gates is not None, 'Error: experiment needs to have opt gates set!'
    assert exp.pmap.opt_map, 'Error: PMAP needs to have opt map set!'

    set_amps_to_zero(exp)

    exp.compute_propagators()
    init_fid = calc_exp_fid(exp, opt_q_inds, opt_q_dims)
    F_bar = init_fid * init_F_bar_mul
    mu = init_mu

    fids = [init_fid]
    # TODO - find better exit conditions
    while (prev_fid := fids[-1]) < 0.999:
        exp_opt = setup_experiment_opt_ctrl(exp)
        run_log_barrier_opt_iter(exp_opt, mu, F_bar)

        exp.compute_propagators()
        fid = calc_exp_fid(exp, opt_q_inds, opt_q_dims)

        if verbose:
            print(f'Fidelity: {fid:.3f}')
            print(f'F_bar:    {F_bar:.3f}')
            print(f'mu:       {mu:.2e}')
            exp.pmap.print_parameters()

        mu *= alpha
        if fid >= prev_fid:
            F_bar = 0.5 * (fid + min(prev_fid, F_bar))
        fids.append(fid)


exp.set_opt_gates([gate_name])
gate_seq = [gate_name]

opt_gates = [gate_name]

gateset_opt_map = [
    [
        ("rx90p[0]", "d1", "carrier", "framechange"),
    ],
    [
        ("rx90p[0]", "d1", "gauss", "amp"),
    ],
    [
        ("rx90p[0]", "d1", "gauss", "freq_offset"),
    ],
]

parameter_map.set_opt_map(gateset_opt_map)

signal_opt_via_log_barrier(exp, opt_q_inds=[0], opt_q_dims=[3])


def plot_dynamics(exp, psi_init, seq, goal=-1):
    """
    Plotting code for time-resolved populations.

    Parameters
    ----------
    psi_init: tf.Tensor
        Initial state or density matrix.
    seq: list
        List of operations to apply to the initial state.
    goal: tf.float64
        Value of the goal function, if used.
    debug: boolean
        If true, return a matplotlib figure instead of saving.
    """
    model = exp.pmap.model
    exp.compute_propagators()
    dUs = exp.partial_propagators
    psi_t = psi_init.numpy()
    pop_t = exp.populations(psi_t, model.lindbladian)
    for gate in seq:
        for du in dUs[gate]:
            psi_t = np.matmul(du.numpy(), psi_t)
            pops = exp.populations(psi_t, model.lindbladian)
            pop_t = np.append(pop_t, pops, axis=1)

    fig, axs = plt.subplots(1, 1)
    ts = exp.ts
    dt = ts[1] - ts[0]
    ts = np.linspace(0.0, dt * pop_t.shape[1], pop_t.shape[1]) / 1e-9
    axs.plot(ts, pop_t.T)
    axs.grid(linestyle="--")
    axs.tick_params(
        direction="in", left=True, right=True, top=True, bottom=True
    )
    axs.set_xlabel('Time [ns]')
    axs.set_ylabel('Population')

    for state_pop in pops:
        state_pop = state_pop.numpy()[0]
        axs.annotate(f'{state_pop * 100:.0f}%', (ts.max(), state_pop))

    plt.legend(model.state_labels)
    plt.xlim(0, ts.max() * 1.1)


psi_init = [[0] * qubit_lvls]
psi_init[0][1] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
plot_dynamics(exp, init_state, [gate_name])

psi_init[0][1] = 0
psi_init[0][0] = 1
init_state = tf.transpose(tf.constant(psi_init, tf.complex128))
plot_dynamics(exp, init_state, [gate_name])

plt.show()
