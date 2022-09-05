import itertools
import os
import tempfile
from collections import defaultdict
from copy import deepcopy
from typing import List

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from c3.c3objs import Quantity
from c3.experiment import Experiment
from c3.generator.devices import AWG
from c3.libraries import algorithms
from c3.libraries.envelopes import envelopes
from c3.libraries.fidelities import unitary_infid_set
from c3.optimizers.optimalcontrol import OptimalControl
from c3.signal.gates import Instruction
from c3.signal.pulse import Envelope
from playground.plot_utils import wait_for_not_mouse_press

SIDEBAND = 50e6

__shared_params = {'amp', 'xy_angle', 'freq_offset', 't_final'}
ENVELOPES_OPT_PARAMS = {'gaussian_nonorm': {'sigma'}}
for env_params in ENVELOPES_OPT_PARAMS.values():
    for shared_param in __shared_params:
        env_params.add(shared_param)


def setup_experiment_opt_ctrl(exp: Experiment, maxiter: int = 50) -> OptimalControl:
    log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

    opt = OptimalControl(
        dir_path=log_dir,
        fid_func=unitary_infid_set,
        fid_subspace=["Q1"],  # TODO-set this automatically
        pmap=exp.pmap,
        algorithm=algorithms.lbfgs,
        options={'maxiter': maxiter},
    )

    opt.set_exp(exp)

    return opt


def calc_exp_fid(exp: Experiment, index: list, dims: list) -> float:
    return (1 - unitary_infid_set(exp.propagators, exp.pmap.instructions, index, dims)).numpy()


def plot_dynamics(exp, psi_init, seq, disp_legend: bool = False):
    """
    Plotting code for time-resolved populations.

    Parameters
    ----------
    psi_init: tf.Tensor
        Initial state or density matrix.
    seq: list
        List of operations to apply to the initial state.
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
    ts = np.linspace(0.0, dt * pop_t.shape[1], pop_t.shape[1])
    axs.plot(ts / 1e-9, pop_t.T)
    axs.grid(linestyle="--")
    axs.tick_params(
        direction="in", left=True, right=True, top=True, bottom=True
    )
    axs.set_xlabel('Time [ns]')
    axs.set_ylabel('Population')
    if disp_legend:
        plt.legend(model.state_labels, loc='center right')


def plot_signal(drivers_signals: dict, t_final: float, awg: AWG, n_points: int = 1000, disp_legend: bool = False):
    signal_t = np.linspace(0, t_final, n_points)

    n_drivers = len(drivers_signals)
    res_signal = np.zeros((n_drivers, len(signal_t)), dtype=np.complex128)
    fig, axs = plt.subplots(1, n_drivers, sharey="all")

    for driver, envs in drivers_signals.items():
        driver_ind = int(driver[1:]) - 1
        for env in envs:
            # de-normalize the amplitude so it has units of volts
            ts = awg.create_ts(0, t_final)
            env.normalize_pulse = False
            area = abs(env.get_shape_values(ts).numpy()).sum()
            amplitude = env.params['amp'].numpy()
            real_amplitude = amplitude / area

            signal = env.get_shape_values(signal_t).numpy() * real_amplitude
            res_signal[driver_ind] += signal

            env.normalize_pulse = True

            axs[driver_ind].plot(signal_t * 1e9, np.real(signal), label=env.name)

    for driver in range(n_drivers):
        ax = axs[driver]

        ax.plot(signal_t * 1e9, np.real(res_signal[driver]),
                label='Resulting signal', c='k', linewidth=2, linestyle='dashed')

        ax.set_title(f'Driver {driver + 1}')
        ax.grid()
        ax.set_xlabel('Time[ns]')
        ax.set_ylabel('Signal[V]')
        ax.set_xlim(0, t_final * 1e9)
        if disp_legend:
            ax.legend()

    plt.suptitle('Real part of pulses')


def get_qubits_population(population: np.array, dims: List[int]) -> np.array:
    """
    Splits the population of all levels of a system into the populations of levels per subsystem.
    Parameters
    ----------
    population: np.array
        The time dependent population of each energy level. First dimension: level index, second dimension: time.
    dims: List[int]
        The number of levels for each subsystem.
    Returns
    -------
    np.array
        The time-dependent population of energy levels for each subsystem. First dimension: subsystem index, second
        dimension: level index, third dimension: time.
    """
    numQubits = len(dims)

    # create a list of all levels
    qubit_levels = []
    for dim in dims:
        qubit_levels.append(list(range(dim)))
    combined_levels = list(itertools.product(*qubit_levels))

    # calculate populations
    qubitsPopulations = np.zeros((numQubits, dims[0], population.shape[1]))
    for idx, levels in enumerate(combined_levels):
        for i in range(numQubits):
            qubitsPopulations[i, levels[i]] += population[idx]
    return qubitsPopulations


def plot_splitted_population(exp: Experiment, psi_init: tf.Tensor, sequence: List[str]) -> None:
    """
    Plots time dependent populations for multiple qubits in separate plots.
    Parameters
    ----------
    exp: Experiment
        The experiment containing the model and propagators
    psi_init: np.array
        Initial state vector
    sequence: List[str]
        List of gate names that will be applied to the state
    -------
    """
    # calculate the time dependent level population
    model = exp.pmap.model
    dUs = exp.partial_propagators
    psi_t = psi_init.numpy()
    pop_t = exp.populations(psi_t, model.lindbladian)
    for gate in sequence:
        for du in dUs[gate]:
            psi_t = np.matmul(du, psi_t)
            pops = exp.populations(psi_t, model.lindbladian)
            pop_t = np.append(pop_t, pops, axis=1)
    dims = [s.hilbert_dim for s in model.subsystems.values()]
    splitted = get_qubits_population(pop_t, dims)

    # timestamps
    dt = exp.ts[1] - exp.ts[0]
    ts = np.linspace(0.0, dt * pop_t.shape[1], pop_t.shape[1])

    # create both subplots
    titles = list(exp.pmap.model.subsystems.keys())
    fig, axs = plt.subplots(1, len(splitted), sharey="all")
    for idx, ax in enumerate(axs):
        ax.plot(ts / 1e-9, splitted[idx].T)
        ax.tick_params(direction="in", left=True, right=True, top=False, bottom=True)
        ax.set_xlabel("Time [ns]")
        ax.set_ylabel("Population")
        ax.set_title(titles[idx])
        ax.legend([str(x) for x in np.arange(dims[idx])])
        ax.grid()

    plt.tight_layout()
    plt.show()


def get_params_dict(opt_params: set, t_final: float) -> dict:
    def_params = {
        'amp': Quantity(value=0., min_val=0.0, max_val=100.0, unit="V"),
        't_final': Quantity(value=t_final, min_val=0.0 * t_final, max_val=2.5 * t_final, unit="s"),
        'xy_angle': Quantity(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit='rad'),
        'freq_offset': Quantity(value=-SIDEBAND - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit='Hz 2pi'),
        'delta': Quantity(value=-1, min_val=-5, max_val=3, unit=""),
        'sigma': Quantity(value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"),
        'risefall': Quantity(value=t_final / 6, min_val=t_final / 10, max_val=t_final / 2, unit="s")
    }

    params = {}
    for param_name in opt_params & set(def_params.keys()):
        params[param_name] = def_params[param_name]

    assert len(params) == len(opt_params), 'Error: different number of params than required!'

    return params


# assumes that the experiment comes with the various devices set up. TODO - make a function that does this
def find_opt_env_for_gate(exp: Experiment, gate: Instruction, plot: bool = False):
    # plan:
    # optimize gate for all driver-envelope combinations
    # if the optimization didn't succeed because of a bad reason - eg. the amplitude reached its upper limit,
    #   not enough iterations, etc., rerun it with looser limits
    # specifically for the amplitude - always initialize to zero and start with very high amplitude,
    #   then successively lower based on previous optimization's result as much as possible until the fidelity is
    #   really bad. Cache good solutions
    best_fid_per_env = {}
    best_params_per_env = {}
    gate_name = gate.get_key()
    drivers = set(exp.pmap.instructions[gate_name].comps.keys())
    n_qubits = len(drivers)
    t_final = gate.t_end
    for env_name, env_to_opt_params in ENVELOPES_OPT_PARAMS.items():
        envelope_func = envelopes[env_name]
        for driver in drivers:
            params = get_params_dict(env_to_opt_params, t_final)
            env = Envelope(name=env_name, normalize_pulse=True, params=params, shape=envelope_func)

            single_env_gate = deepcopy(gate)
            single_env_gate.add_component(env, driver)
            exp.pmap.instructions = {gate_name: single_env_gate}

            opt_params = get_opt_params_conf(driver, env_to_opt_params)
            exp.pmap.set_opt_map(opt_params)
            exp.pmap.update_parameters()

            best_sig_fid, opt_params = opt_single_sig_exp(exp)

            best_fid_per_env[env_name] = best_sig_fid
            best_params_per_env[env_name] = opt_params

            if plot:
                # TODO - add more plotting functionality
                psi_init = get_init_state([0] * n_qubits, exp)
                plot_dynamics(exp, psi_init, [gate_name])

                wait_for_not_mouse_press()

            exp.pmap.instructions = {}


if __name__ == '__main__':
    ?
    find_opt_env_for_gate(?, ?, plot = True)
