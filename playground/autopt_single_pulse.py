import itertools
import os
import tempfile
from copy import deepcopy
from functools import reduce
from typing import List

import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

from c3.c3objs import Quantity
from c3.experiment import Experiment
import c3.generator.devices as devices
from c3.generator.generator import Generator
from c3.libraries import algorithms, hamiltonians, chip
from c3.libraries.envelopes import envelopes
from c3.libraries.fidelities import unitary_infid_set
from c3.model import Model
from c3.optimizers.optimalcontrol import OptimalControl
from c3.parametermap import ParameterMap
from c3.signal import gates, pulse
from c3.signal.gates import Instruction
from c3.signal.pulse import Envelope
from playground.plot_utils import wait_for_not_mouse_press

SIDEBAND = 50e6

__shared_params = {'amp', 'xy_angle', 'freq_offset', 't_final', 'delta'}
ENVELOPES_OPT_PARAMS = {'gaussian_nonorm': {'sigma'}, 'hann': set(), 'blackman_window': set(),
                        'flattop_risefall': {'risefall'}}
for env_params in ENVELOPES_OPT_PARAMS.values():
    for shared_param in __shared_params:
        env_params.add(shared_param)


def setup_experiment_opt_ctrl(exp: Experiment, maxiter: int = 50) -> OptimalControl:
    # TODO - better set this
    n_qubits = len(exp.pmap.model.dims)
    fid_subspace = [f'Q{i + 1}' for i in range(n_qubits)]
    log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

    opt = OptimalControl(
        dir_path=log_dir,
        fid_func=unitary_infid_set,
        fid_subspace=fid_subspace,
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

    ts = exp.ts
    dt = ts[1] - ts[0]
    ts = np.linspace(0.0, dt * pop_t.shape[1], pop_t.shape[1])
    plt.plot(ts / 1e-9, pop_t.T)
    plt.grid(linestyle="--")
    plt.tick_params(
        direction="in", left=True, right=True, top=True, bottom=True
    )
    plt.xlabel('Time [ns]')
    plt.ylabel('Population')
    if disp_legend:
        plt.legend(model.state_labels, loc='center right')


def plot_signal(drivers_signals: dict, t_final: float, awg: devices.AWG, n_points: int = 1000,
                disp_legend: bool = False):
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
        'amp': Quantity(value=1e-5, min_val=0.0, max_val=500., unit="V"),
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


def get_opt_params_conf(driver: str, gate_key: str, env_name, env_to_opt_params: set) -> list:
    PARAMS_TO_EXCLUDE = {'t_final', 'sigma'}
    return [[(gate_key, driver, env_name, param), ] for param in env_to_opt_params - PARAMS_TO_EXCLUDE]


def get_init_state(exp: Experiment, energy_level: int = None) -> tf.Tensor:
    dims = exp.pmap.model.dims
    n_lvls = reduce(lambda x, y: x * y, dims)

    if energy_level is None:
        energy_level = 0

    psi_init = [[0] * n_lvls]
    psi_init[0][energy_level] = 1
    return tf.transpose(tf.constant(psi_init, tf.complex128))


def opt_single_sig_exp(exp: Experiment) -> tuple:
    exp_opt = setup_experiment_opt_ctrl(exp)
    exp_opt.optimize_controls()

    best_score = exp_opt.current_best_goal
    best_params = exp_opt.current_best_params

    return best_score, best_params


def find_opt_params_for_single_env(exp: Experiment, amp: Quantity, driver: str = None, env_name: str = None,
                                   gate_name: str = None, debug: bool = False, MIN_AMP: float = 5,
                                   AMP_RED_FCTR: float = 0.5, MAX_PLOT_SCORE: float = 0.2) -> tuple:
    best_overall_score = np.inf
    best_overall_params = None
    while (max_amp := amp.get_limits()[1]) > MIN_AMP:
        amp.set_value(max_amp / 2)

        best_score, best_params_vals = opt_single_sig_exp(exp)

        if best_score < best_overall_score:
            best_overall_score = best_score
            best_overall_params = best_params_vals

        if debug:
            print(f'Driver:   {driver}')
            print(f'Envelope: {env_name}')
            print(f'Score: {best_score:.3f}')
            print(f'Amplitude:{amp.get_value():.1f}')
            print(f'Max amp.: {max_amp:.1f}')

            if best_score < MAX_PLOT_SCORE:
                # TODO - add more plotting functionality
                psi_init = get_init_state(exp)
                plot_dynamics(exp, psi_init, [gate_name])
                plt.title(f'{driver}-{env_name}, F={best_score:.3f}')

                wait_for_not_mouse_press()

                plt.clf()

        amp._set_limits(0, max_amp * AMP_RED_FCTR)

    return best_overall_score, best_overall_params


def get_carrier_opt_params(drivers: set, gate_name: str) -> list:
    carrier_opt_params = {'framechange'}
    return [[(gate_name, driver, 'carrier', carr_param)] for driver in drivers for carr_param in carrier_opt_params]


def get_cached_exp_path(cache_dir: str, driver: str, env_name: str) -> str:
    return os.path.join(cache_dir, f'best_{driver}_{env_name}.hjson')


# assumes that the experiment comes with the various devices set up. TODO - make a function that does this
def find_opt_env_for_gate(exp: Experiment, gate: Instruction, base_opt_params: list, cache_dir: str,
                          debug: bool = False):
    gate_name = gate.get_key()
    drivers = set(exp.pmap.instructions[gate_name].comps.keys())
    best_score_per_env = {driver: {} for driver in drivers}
    best_params_per_env = {driver: {} for driver in drivers}
    opt_map_params_per_env = {driver: {} for driver in drivers}
    t_final = gate.t_end

    for env_name, env_to_opt_params in ENVELOPES_OPT_PARAMS.items():
        envelope_func = envelopes[env_name]
        for driver in drivers:
            params = get_params_dict(env_to_opt_params, t_final)
            env = Envelope(name=env_name, normalize_pulse=True, params=params, shape=envelope_func)

            gate_with_extra_env = deepcopy(gate)
            gate_with_extra_env.add_component(env, driver)
            base_instructions = deepcopy(exp.pmap.instructions)
            exp.pmap.instructions.update({gate_name: gate_with_extra_env})

            opt_params = get_opt_params_conf(driver, gate_name, env_name, env_to_opt_params) + base_opt_params
            exp.pmap.update_parameters()
            exp.pmap.set_opt_map(opt_params)

            amp = params['amp']
            best_score, best_params_vals = find_opt_params_for_single_env(exp, amp, driver, env_name, gate_name, debug)

            best_score_per_env[driver][env_name] = best_score
            best_params_per_env[driver][env_name] = best_params_vals
            opt_map_params_per_env[driver][env_name] = opt_params

            exp.pmap.set_parameters(best_params_vals, extend_bounds=True)
            exp.pmap.update_parameters()
            exp.compute_propagators()
            cache_path = get_cached_exp_path(cache_dir, driver, env_name)
            exp.write_config(cache_path)

            exp.pmap.instructions = base_instructions

    return best_score_per_env, best_params_per_env, opt_map_params_per_env


def optimize_gate(exp: Experiment, gate: Instruction, cache_dir: str, opt_map_params: list = None,
                  n_pulses_to_add: int = 1, MAX_SCORE_CONTINUE_RECURSION: float = 0.2, debug: bool = False):
    if opt_map_params is None:
        gate_name = gate.get_key()
        drivers = set(exp.pmap.instructions[gate_name].comps.keys())
        opt_map_params = get_carrier_opt_params(drivers, gate_name)

    score_per_env, params_per_env, opt_map_params_per_env = find_opt_env_for_gate(exp, gate, opt_map_params, cache_dir,
                                                                                  debug=debug)

    if n_pulses_to_add == 1:
        return

    for driver, env_scores in score_per_env.items():
        for env_name, env_score in env_scores.items():
            if debug:
                print(f'{env_name} score: {env_score:.3f}, need to add {n_pulses_to_add - 1} pulses')

            if env_score > MAX_SCORE_CONTINUE_RECURSION:
                continue

            pulse_cache_dir = os.path.join(cache_dir, f'{driver}_{env_name}')
            os.mkdir(pulse_cache_dir)

            pulse_exp = Experiment()
            pulse_exp.read_config(get_cached_exp_path(cache_dir, driver, env_name))
            pulse_gate = exp.pmap.instructions[gate.get_key()]
            existing_opt_map_params = deepcopy(opt_map_params_per_env[driver][env_name])
            optimize_gate(pulse_exp, pulse_gate, pulse_cache_dir, existing_opt_map_params, n_pulses_to_add - 1,
                          debug=debug)


if __name__ == '__main__':
    qubit_lvls = 3
    freq_q1 = 5e9
    anhar_q1 = -210e6
    t1_q1 = 27e-6
    t2star_q1 = 39e-6
    qubit_temp = 50e-3

    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Quantity(value=freq_q1, min_val=4.995e9, max_val=5.005e9, unit="Hz 2pi"),
        anhar=Quantity(value=anhar_q1, min_val=-380e6, max_val=-20e6, unit="Hz 2pi"),
        hilbert_dim=qubit_lvls,
        t1=Quantity(value=t1_q1, min_val=1e-6, max_val=90e-6, unit="s"),
        t2star=Quantity(value=t2star_q1, min_val=10e-6, max_val=90e-3, unit="s"),
        temp=Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    )

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
    q1q2 = chip.Coupling(
        name="Q1-Q2",
        desc="coupling",
        comment="Coupling qubit 1 to qubit 2",
        connected=["Q1", "Q2"],
        strength=Quantity(
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

    model = Model(
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

    generator = Generator(
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

    __t_final = 45e-9  # Time for two qubit gates

    lo_freq_q1 = freq_q1 + SIDEBAND
    lo_freq_q2 = freq_q2 + SIDEBAND

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

    cnot12 = gates.Instruction(
        name="cnot", targets=[0, 1], t_start=0.0, t_end=__t_final, channels=["d1", "d2"],
        ideal=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ])
    )

    cz = gates.Instruction(
        name="cz", targets=[0, 1], t_start=0.0, t_end=__t_final, channels=["d1", "d2"],
        ideal=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ])
    )

    cy = gates.Instruction(
        name="cy", targets=[0, 1], t_start=0.0, t_end=__t_final, channels=["d1", "d2"],
        ideal=np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ])
    )

    swap = gates.Instruction(
        name="swap", targets=[0, 1], t_start=0.0, t_end=__t_final, channels=["d1", "d2"],
        ideal=np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ])
    )

    gate = cnot12

    gate.add_component(carr_2Q_1, "d1")
    gate.add_component(carr_2Q_2, "d2")
    gate.comps["d1"]["carrier"].params["framechange"].set_value(
        (-SIDEBAND * __t_final) * 2 * np.pi % (2 * np.pi)
    )

    parameter_map = ParameterMap(instructions=[gate], model=model, generator=generator)
    exp = Experiment(pmap=parameter_map)

    optimize_gate(exp, gate, cache_dir='autopt_cache', n_pulses_to_add=2, debug=True)
