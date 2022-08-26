import itertools
from typing import List

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from c3.experiment import Experiment
from c3.generator.devices import AWG


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


def plot_signal(awg: AWG, drivers_signals: dict, t_final: float,
                n_points: int = 1000, disp_legend: bool = False):
    signal_t = np.linspace(0, t_final, n_points)

    n_drivers = len(drivers_signals)
    res_signal = np.zeros((n_drivers, len(signal_t)), dtype=np.complex128)
    fig, axs = plt.subplots(1, n_drivers, sharey="all")

    for driver, envs in drivers_signals.items():
        driver_ind = int(driver[1:]) - 1
        for env in envs.values():
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


def plot_splitted_population(
        exp: Experiment,
        psi_init: tf.Tensor,
        sequence: List[str]
) -> None:
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


def wait_for_not_mouse_press():
    while not plt.waitforbuttonpress():
        pass
