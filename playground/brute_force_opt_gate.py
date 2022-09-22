import itertools
import os
import pickle
import tempfile
from copy import deepcopy
from typing import List, Callable

import numpy as np
from matplotlib import pyplot as plt

import c3.generator.devices as devices
from alex_system.four_level_transmons import custom_gates
from alex_system.four_level_transmons.DataOutput import DataOutput
from alex_system.four_level_transmons.utilities import createQubits, createChainCouplings, createDrives, \
    createGenerator2LOs
from c3.c3objs import Quantity
from c3.experiment import Experiment
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
from playground.plot_utils import wait_for_not_mouse_press, plot_dynamics, get_init_state

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


def calc_exp_fid(exp: Experiment) -> float:
    dims = exp.pmap.model.dims
    index = list(range(len(dims)))
    return (1 - unitary_infid_set(exp.propagators, exp.pmap.instructions, index, dims)).numpy()


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


def opt_single_sig_exp(exp: Experiment) -> tuple:
    exp_opt = setup_experiment_opt_ctrl(exp)
    exp_opt.optimize_controls()

    best_infid = exp_opt.current_best_goal
    best_params = exp_opt.current_best_params

    return best_infid, best_params


def find_opt_params_for_single_env(exp: Experiment, amp: Quantity, cache_path: str, driver: str = None,
                                   env_name: str = None, gate_name: str = None, debug: bool = False,
                                   MAX_PLOT_INFID: float = 0.05, MAX_INFID_TO_CACHE: float = 0.03) -> tuple:
    infid_per_amp = {}
    params_per_amp = {}
    n_cached = 0

    def run_opt():
        nonlocal n_cached

        max_amp = amp.get_limits()[1]

        best_infid, best_params_vals = opt_single_sig_exp(exp)

        infid_per_amp[max_amp] = best_infid
        params_per_amp[max_amp] = best_params_vals

        if debug:
            print(f'Driver:   {driver}')
            print(f'Envelope: {env_name}')
            print(f'Infid.:   {best_infid:.3f}')
            print(f'Amplitude:{amp.get_value():.1f}')
            print(f'Max amp.: {max_amp:.1f}')

            if best_infid < MAX_PLOT_INFID:
                # TODO - add more plotting functionality
                psi_init = get_init_state(exp)
                plot_dynamics(exp, psi_init, [gate_name])
                plt.title(f'{driver}-{env_name}, F={best_infid:.3f}')

                wait_for_not_mouse_press()

                plt.close()

        if best_infid < MAX_INFID_TO_CACHE:
            good_exp_cache_path = cache_path.format(cache_num=n_cached)
            exp.write_config(good_exp_cache_path)
            if debug:
                print('Caching to ' + good_exp_cache_path)
                print(f'Infid={best_infid:.3f}')

            n_cached += 1

    run_for_amps_range(run_opt, amp)

    best_overall_infid = np.inf
    best_overall_params = None
    for amp in infid_per_amp.keys():
        if (infid := infid_per_amp[amp]) < best_overall_infid:
            best_overall_infid = infid
            best_overall_params = params_per_amp[amp]

    return best_overall_infid, best_overall_params


def get_carrier_opt_params(drivers: set, gate_name: str) -> list:
    carrier_opt_params = {'framechange'}
    return [[(gate_name, driver, 'carrier', carr_param)] for driver in drivers for carr_param in carrier_opt_params]


def get_cached_exp_path(cache_dir: str, driver: str, env_name: str) -> str:
    return os.path.join(cache_dir, f'best_{driver}_{env_name}.hjson')


def cache_exp(exp: Experiment, cache_dir: str, driver, env_name: str, params_vals: list = None):
    if params_vals is not None:
        exp.pmap.set_parameters(params_vals, extend_bounds=True)

    exp.pmap.update_parameters()
    exp.compute_propagators()
    cache_path = get_cached_exp_path(cache_dir, driver, env_name)
    exp.write_config(cache_path)


def run_for_amps_range(func: Callable, amp, MIN_AMP: float = 5, AMP_RED_FCTR: float = 0.5, ):
    while (max_amp := amp.get_limits()[1]) > MIN_AMP:
        amp.set_value(max_amp / 2)

        func()

        amp._set_limits(0, max_amp * AMP_RED_FCTR)


# assumes that the experiment comes with the various devices set up. TODO - make a function that does this
def find_opt_env_for_gate(exp: Experiment, gate: Instruction, base_opt_params: list, cache_dir: str,
                          n_pulses_to_opt: int = 1, pulse_suffix: str = '', debug: bool = False):
    gate_name = gate.get_key()
    drivers = set(exp.pmap.instructions[gate_name].comps.keys())
    best_infid_per_env = {driver: {} for driver in drivers}
    best_params_per_env = {driver: {} for driver in drivers}
    opt_map_params_per_env = {driver: {} for driver in drivers}
    t_final = gate.t_end

    for env_name, env_to_opt_params in ENVELOPES_OPT_PARAMS.items():
        envelope_func = envelopes[env_name]
        env_name = env_name + pulse_suffix
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
            if n_pulses_to_opt > 1:
                def recursive_opt_for_different_amps():
                    max_amp = amp.get_limits()[1]
                    subcache_dir = os.path.join(cache_dir, env_name + f'_{driver}_max_amp={max_amp:.1f}_sub_opt')
                    if not os.path.isdir(subcache_dir):
                        os.mkdir(subcache_dir)

                    if debug:
                        print(f'Preparing subopt routine, caching to {subcache_dir}')

                    pulse_suffix = str(n_pulses_to_opt - 1)
                    find_opt_env_for_gate(exp, gate_with_extra_env, opt_params, subcache_dir, n_pulses_to_opt - 1,
                                          pulse_suffix=pulse_suffix, debug=debug)

                run_for_amps_range(recursive_opt_for_different_amps, amp)
            else:
                good_res_cache_dir = os.path.join(cache_dir, 'good_infid_exps')
                if not os.path.isdir(good_res_cache_dir):
                    os.mkdir(good_res_cache_dir)
                good_res_cache_path = os.path.join(good_res_cache_dir, f'{driver}_{env_name}_' + '{cache_num}.hjson')
                best_infid, best_params_vals = find_opt_params_for_single_env(exp, amp, good_res_cache_path, driver,
                                                                              env_name, gate_name, debug)

                best_infid_per_env[driver][env_name] = best_infid
                best_params_per_env[driver][env_name] = best_params_vals
                opt_map_params_per_env[driver][env_name] = opt_params

                cache_exp(exp, cache_dir, driver, env_name, best_params_vals)

            exp.pmap.instructions = base_instructions

    return best_infid_per_env, best_params_per_env, opt_map_params_per_env


def cache_opt_map_params(opt_map_params: list, cache_dir: str):
    cache_path = os.path.join(cache_dir, 'base_opt_map_params.pkl')
    with open(cache_path, 'wb') as file:
        pickle.dump(opt_map_params, file)


def read_cached_opt_map_params(cache_dir: str) -> list:
    cache_path = os.path.join(cache_dir, 'base_opt_map_params.pkl')
    with open(cache_path, 'rb') as file:
        return pickle.load(file)


def optimize_gate(exp: Experiment, gate: Instruction, cache_dir: str, opt_map_params: list = None,
                  n_pulses_to_add: int = 1, opt_all_at_once: bool = False, MAX_INFID_CONTINUE_RECURSION: float = 0.2,
                  debug: bool = False):
    if opt_map_params is None:
        gate_name = gate.get_key()
        drivers = set(exp.pmap.instructions[gate_name].comps.keys())
        opt_map_params = get_carrier_opt_params(drivers, gate_name)

    n_pulses_to_opt = 1 if not opt_all_at_once else n_pulses_to_add
    infid_per_env, params_per_env, opt_map_params_per_env = find_opt_env_for_gate(exp, gate, opt_map_params, cache_dir,
                                                                                  n_pulses_to_opt=n_pulses_to_opt,
                                                                                  pulse_suffix=str(n_pulses_to_add),
                                                                                  debug=debug)

    if n_pulses_to_add == 1 or opt_all_at_once:
        return

    for driver, env_scores in infid_per_env.items():
        for env_name, env_score in env_scores.items():
            if env_score > MAX_INFID_CONTINUE_RECURSION:
                continue

            if debug:
                print(f'{env_name} on driver {driver} infidelity: {env_score:.3f}, '
                      f'need to add {n_pulses_to_add - 1} pulses')

            pulse_cache_dir = os.path.join(cache_dir, f'{driver}_{env_name}')
            if not os.path.isdir(pulse_cache_dir):
                os.mkdir(pulse_cache_dir)

            pulse_exp = Experiment()
            pulse_exp.read_config(get_cached_exp_path(cache_dir, driver, env_name))
            pulse_gate = pulse_exp.pmap.instructions[gate.get_key()]
            existing_opt_map_params = deepcopy(opt_map_params_per_env[driver][env_name])
            cache_opt_map_params(existing_opt_map_params, pulse_cache_dir)
            optimize_gate(pulse_exp, pulse_gate, pulse_cache_dir, existing_opt_map_params, n_pulses_to_add - 1,
                          debug=debug)


def get_ccx_system(t_final=100e-9, qubit_lvls=4):
    freq_q1 = 5e9
    anhar_q1 = -210e6
    t1_q1 = 27e-6
    t2star_q1 = 39e-6
    qubit_temp = 50e-3

    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Quantity(value=freq_q1, min_val=4.995e9, max_val=5.005e9, unit='Hz 2pi'),
        anhar=Quantity(value=anhar_q1, min_val=-380e6, max_val=-120e6, unit='Hz 2pi'),
        hilbert_dim=qubit_lvls,
        t1=Quantity(value=t1_q1, min_val=1e-6, max_val=90e-6, unit='s'),
        t2star=Quantity(value=t2star_q1, min_val=10e-6, max_val=90e-3, unit='s'),
        temp=Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit='K')
    )

    freq_q2 = 5.3e9
    anhar_q2 = -240e6
    t1_q2 = 23e-6
    t2star_q2 = 31e-6
    q2 = chip.Qubit(
        name="Q2",
        desc="Qubit 2",
        freq=Quantity(value=freq_q2, min_val=5.295e9, max_val=5.305e9, unit='Hz 2pi'),
        anhar=Quantity(value=anhar_q2, min_val=-380e6, max_val=-120e6, unit='Hz 2pi'),
        hilbert_dim=qubit_lvls,
        t1=Quantity(value=t1_q2, min_val=1e-6, max_val=90e-6, unit='s'),
        t2star=Quantity(value=t2star_q2, min_val=10e-6, max_val=90e-6, unit='s'),
        temp=Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit='K')
    )

    freq_q3 = 5.6e9
    anhar_q3 = -240e6
    t1_q3 = 23e-6
    t2star_q3 = 31e-6
    q3 = chip.Qubit(
        name="Q3",
        desc="Qubit 3",
        freq=Quantity(value=freq_q3, min_val=5.595e9, max_val=5.605e9, unit='Hz 2pi'),
        anhar=Quantity(value=anhar_q3, min_val=-380e6, max_val=-120e6, unit='Hz 2pi'),
        hilbert_dim=qubit_lvls,
        t1=Quantity(value=t1_q3, min_val=1e-6, max_val=90e-6, unit='s'),
        t2star=Quantity(value=t2star_q3, min_val=10e-6, max_val=90e-6, unit='s'),
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

    q1q3 = chip.Coupling(
        name="Q1-Q3",
        desc="coupling",
        comment="Coupling qubit 1 to qubit 3",
        connected=["Q1", "Q3"],
        strength=Quantity(
            value=coupling_strength,
            min_val=-1 * 1e3,
            max_val=200e6,
            unit='Hz 2pi'
        ),
        hamiltonian_func=hamiltonians.int_XX
    )

    q2q3 = chip.Coupling(
        name="Q2-Q3",
        desc="coupling",
        comment="Coupling qubit 2 to qubit 3",
        connected=["Q2", "Q3"],
        strength=Quantity(
            value=coupling_strength,
            min_val=-1 * 1e3,
            max_val=200e6,
            unit='Hz 2pi'
        ),
        hamiltonian_func=hamiltonians.int_XX
    )

    drive1 = chip.Drive(
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

    drive3 = chip.Drive(
        name="d3",
        desc="Drive 3",
        comment="Drive line 3 on qubit 3",
        connected=["Q3"],
        hamiltonian_func=hamiltonians.x_drive
    )

    model = Model(
        [q1, q2, q3],  # Individual, self-contained components
        [drive1, drive2, drive3, q1q2, q1q3, q2q3],  # Interactions between components
    )
    model.set_lindbladian(False)
    model.set_dressed(True)

    sim_res = 100e9  # Resolution for numerical simulation
    awg_res = 2e9  # Realistic, limited resolution of an AWG
    v2hz = 1e9

    lo = devices.LO(name='lo', resolution=sim_res)
    awg = devices.AWG(name='awg', resolution=awg_res)
    mixer = devices.Mixer(name='mixer')
    dig_to_an = devices.DigitalToAnalog(name="dac", resolution=sim_res)
    v_to_hz = devices.VoltsToHertz(
        name='v_to_hz',
        V_to_Hz=Quantity(value=v2hz, min_val=0.9e9, max_val=1.1e9, unit='Hz/V')
    )

    generator = Generator(
        devices={
            "LO": lo,
            "AWG": awg,
            "DigitalToAnalog": dig_to_an,
            "Mixer": mixer,
            "VoltsToHertz": v_to_hz
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
            "d3": {
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Mixer": ["LO", "DigitalToAnalog"],
                "VoltsToHertz": ["Mixer"],
            }
        }
    )

    lo_freq_q1 = freq_q1 + SIDEBAND
    lo_freq_q2 = freq_q2 + SIDEBAND
    lo_freq_q3 = freq_q3 + SIDEBAND

    carr_3Q_1 = pulse.Carrier(
        name="carrier",
        desc="Carrier on drive 1",
        params={
            'freq': Quantity(value=lo_freq_q3, min_val=0.8 * lo_freq_q1, max_val=1.2 * lo_freq_q3, unit='Hz 2pi'),
            'framechange': Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit='rad')
        }
    )

    carr_3Q_2 = pulse.Carrier(
        name="carrier",
        desc="Carrier on drive 2",
        params={
            'freq': Quantity(value=lo_freq_q3, min_val=0.8 * lo_freq_q1, max_val=1.2 * lo_freq_q3, unit='Hz 2pi'),
            'framechange': Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit='rad')
        }
    )

    carr_3Q_3 = pulse.Carrier(
        name="carrier",
        desc="Carrier on drive 3",
        params={
            'freq': Quantity(value=lo_freq_q3, min_val=0.8 * lo_freq_q1, max_val=1.2 * lo_freq_q3, unit='Hz 2pi'),
            'framechange': Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit='rad')
        }
    )

    ccnot = gates.Instruction(
        name="ccnot", targets=[0, 1, 2], t_start=0.0, t_end=t_final, channels=["d1", "d2", "d3"],
        ideal=np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ])
    )

    gate = ccnot
    dir = f'toffoli_{t_final * 1e9:.0f}ns_trial'

    gate.add_component(carr_3Q_1, "d1")
    gate.add_component(carr_3Q_2, "d2")
    gate.add_component(carr_3Q_3, "d3")
    gate.comps["d1"]["carrier"].params["framechange"].set_value(
        (-SIDEBAND * t_final) * 2 * np.pi % (2 * np.pi)
    )
    gate.comps["d2"]["carrier"].params["framechange"].set_value(
        (-SIDEBAND * t_final) * 2 * np.pi % (2 * np.pi)
    )

    return gate, dir, model, generator


def get_alex_system(output_dir='alex_sys_output_dir', t_final=500e-9):
    qubit_levels = [4, 4]
    qubit_frequencies = [5e9, 5e9]
    anharmonicities = [-300e6, -300e6]
    t1s = [25e-6, 25e-6]
    t2stars = [35e-6, 35e-6]
    qubit_temps = 50e-3
    couplingStrength = 20e6
    isDressed = True
    sim_res = 50e9
    awg_res = 2e9

    print("qubits frequencies: ", qubit_frequencies, "anharmonicities: ", anharmonicities,
          "coupling: ", couplingStrength)

    level_labels_transmon = ["|0,0\\rangle", "|0,1\\rangle", "|1,0\\rangle", "|1,1\\rangle"]
    for i in range(len(level_labels_transmon), max(qubit_levels)):
        level_labels_transmon.append("leakage")
    level_labels = []
    level_labels_with_leakage = []
    level_labels_short = []
    for i in range(qubit_levels[0]):
        for j in range(qubit_levels[1]):
            if i > 3 or j > 3:
                level_labels_with_leakage.append("leakage")
                level_labels_short.append(None)
            else:
                s = f"${level_labels_transmon[i]},{level_labels_transmon[j]}$"
                level_labels.append(s)
                level_labels_with_leakage.append(s)
                level_labels_short.append(f"{i},{j}")

    qubits = createQubits(qubit_levels, qubit_frequencies, anharmonicities,
                          t1s, t2stars, qubit_temps)
    coupling = createChainCouplings([couplingStrength], qubits)
    drives = createDrives(qubits)

    # Create the model
    model = Model(qubits, coupling + drives)
    model.set_lindbladian(False)
    model.set_dressed(isDressed)
    model.set_FR(False)  # change?

    generator = createGenerator2LOs(drives, sim_res=sim_res, awg_res=awg_res)

    # Envelopes and carriers
    carrier_freqs = [
        [40e6, 563e6],
        [121e6, 644e6]
    ]

    carrier_framechange = [
        [0.01, 0.01],
        [0.01, 0.01]
    ]

    envelopesForDrive = {d.name: [] for d in drives}
    carriers = []
    carriersForDrive = {d.name: [] for d in drives}

    # create carriers and envelopes
    for idx in [0, 1]:
        for i in range(0, len(carrier_freqs[idx])):
            carrier_parameters = {
                "freq": Quantity(value=carrier_freqs[idx][i], min_val=0.98 * carrier_freqs[idx][i],
                                 max_val=1.02 * carrier_freqs[idx][i], unit="Hz 2pi"),
                "framechange": Quantity(value=carrier_framechange[idx][i], min_val=-np.pi, max_val=3 * np.pi,
                                        unit="rad"),
            }
            carrier = pulse.Carrier(
                name=f"carrier_{drives[idx].name}_{i + 1}",
                desc="Frequency of the local oscillator",
                params=carrier_parameters,
            )
            carriers.append(carrier)
            carriersForDrive[drives[idx].name].append(carrier)

    print("carrier: ", [[carrier.params["freq"] for carrier in carriers] for carriers in carriersForDrive.values()])
    print("amp: ", [[env.params["amp"] for env in envelopes] for envelopes in envelopesForDrive.values()])

    ideal_gate = custom_gates.GATE_iSWAP_t1q2_t2q2

    gate = gates.Instruction(
        name="iswap_t1q2_t2q2",
        targets=[0, 1],
        t_start=0.0,
        t_end=t_final,
        channels=[d.name for d in drives],
        ideal=ideal_gate,
    )

    for drive in drives:
        for env in envelopesForDrive[drive.name]:
            gate.add_component(deepcopy(env), drive.name)
        for carrier in carriersForDrive[drive.name]:
            gate.add_component(deepcopy(carrier), drive.name)

    return gate, output_dir, model, generator


def get_2q_system(gate_name: str, qubit_lvls=4, __t_final=45e-9, doubly_resonant: bool = False):
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

    freq_q2 = 5.6e9 if not doubly_resonant else freq_q1
    q2_freq_quantity = Quantity(value=freq_q2, min_val=4.995e9, max_val=5.005e9, unit='Hz 2pi') \
        if not doubly_resonant else deepcopy(q1.freq)
    anhar_q2 = -240e6
    t1_q2 = 23e-6
    t2star_q2 = 31e-6
    q2 = chip.Qubit(name="Q2", desc="Qubit 2",
                    freq=q2_freq_quantity,
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
            "LO": lo,
            "AWG": awg,
            "DigitalToAnalog": dig_to_an,
            "Mixer": mixer,
            "VoltsToHertz": v_to_hz,
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

    comp_gate = gates.Instruction(
        name="comp_gate", targets=[0, 1], t_start=0.0, t_end=__t_final, channels=["d1", "d2"],
        ideal=np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0]
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

    iswap = gates.Instruction(
        name="iswap", targets=[0, 1], t_start=0.0, t_end=__t_final, channels=["d1", "d2"],
        ideal=np.array([
            [1, 0, 0, 0],
            [0, 0, 1j, 0],
            [0, 1j, 0, 0],
            [0, 0, 0, 1]
        ])
    )

    gate = {'iswap': iswap, 'cnot': cnot12, 'cx': cnot12, 'cz': cz, 'cy': cy, 'swap': swap,
            'comp': comp_gate}[gate_name]

    gate.add_component(carr_2Q_1, "d1")
    gate.add_component(carr_2Q_2, "d2")
    gate.comps["d1"]["carrier"].params["framechange"].set_value(
        (-SIDEBAND * __t_final) * 2 * np.pi % (2 * np.pi)
    )

    return gate, model, generator


if __name__ == '__main__':
    gate, model, generator = get_2q_system('cx')
    dir = 'cnot_multipulse_opt'

    # gate, dir, model, generator = get_ccx_system(t_final=100e-9, qubit_lvls=4)

    parameter_map = ParameterMap(instructions=[gate], model=model, generator=generator)
    exp = Experiment(pmap=parameter_map)

    optimize_gate(exp, gate, cache_dir=dir, n_pulses_to_add=2, opt_all_at_once=True, debug=True)
