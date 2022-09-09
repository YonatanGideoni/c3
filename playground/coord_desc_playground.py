import os
import tempfile
from copy import deepcopy
from typing import Tuple

import numpy as np
import pandas as pd
from modAL.acquisition import max_EI
from modAL.models import BayesianOptimizer
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

from c3.c3objs import Quantity
from c3.experiment import Experiment
from c3.generator import devices
from c3.generator.generator import Generator
from c3.libraries import chip, envelopes, hamiltonians
from c3.libraries.algorithms import lbfgs
from c3.libraries.fidelities import sparse_unitary_infid_set, unitary_infid_set
from c3.model import Model
from c3.optimizers.optimalcontrol import OptimalControl
from c3.parametermap import ParameterMap
from c3.signal import pulse, gates
from c3.signal.pulse import Envelope

__shared_params = {'amp', 'xy_angle', 'freq_offset', 't_final', 'delta'}
ENV_PARAMS = {'gaussian_nonorm': {'sigma'}, 'hann': set(), 'blackman_window': set(),
              'flattop_risefall': {'risefall'}}
for env_params in ENV_PARAMS.values():
    for shared_param in __shared_params:
        env_params.add(shared_param)

SIDEBAND = 50e6


def calc_exp_fid(exp: Experiment, index: list) -> float:
    dims = exp.pmap.model.dims
    return (1 - unitary_infid_set(exp.propagators, exp.pmap.instructions, index, dims)).numpy()


def set_point_in_pmap(exp: Experiment, point: np.ndarray):
    exp.pmap.set_parameters_scaled(point)


def setup_rx90_exp(t_final: float):
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

    drive = chip.Drive(
        name="d1",
        desc="Drive 1",
        comment="Drive line 1 on qubit 1",
        connected=["Q1"],
        hamiltonian_func=hamiltonians.x_drive
    )

    model = Model(
        [q1],  # Individual, self-contained components
        [drive],  # Interactions between components
    )

    model.set_lindbladian(False)
    model.set_dressed(True)

    sim_res = 100e9  # Resolution for numerical simulation
    awg_res = 2e9  # Realistic, limited resolution of an AWG
    v2hz = 1e9
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
                V_to_Hz=Quantity(value=v2hz, min_val=0.9 * v2hz, max_val=1.1 * v2hz, unit="Hz/V"),
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

    SIDEBAND = 50e6

    lo_freq_q1 = 5e9 + SIDEBAND
    carrier_parameters = {
        "freq": Quantity(value=lo_freq_q1, min_val=4.5e9, max_val=6e9, unit="Hz 2pi"),
        "framechange": Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
    }
    carr = pulse.Carrier(
        name="carrier", desc="Frequency of the local oscillator", params=carrier_parameters
    )

    rx90p_q1 = gates.Instruction(
        name="rx90p", targets=[0], t_start=0.0, t_end=t_final, channels=["d1"]
    )

    rx90p_q1.add_component(carr, "d1")

    gate = rx90p_q1
    gate_name = gate.get_key()

    parameter_map = ParameterMap(instructions=[gate], model=model, generator=generator)
    exp = Experiment(pmap=parameter_map)

    exp.set_opt_gates([gate_name])

    return exp, gate


def get_opt_params_conf(driver: str, gate_key: str, env_name, env_to_opt_params: set) -> list:
    return [[(gate_key, driver, env_name, param), ] for param in env_to_opt_params]


def get_params_dict(params: set, t_final: float) -> dict:
    def_params = {
        'amp': Quantity(value=1e-5, min_val=0.0, max_val=20., unit="V"),
        't_final': Quantity(value=t_final, min_val=0.5 * t_final, max_val=2.5 * t_final, unit="s"),
        'xy_angle': Quantity(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit='rad'),
        'freq_offset': Quantity(value=-SIDEBAND - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit='Hz 2pi'),
        'delta': Quantity(value=-1, min_val=-5, max_val=3, unit=""),
        'sigma': Quantity(value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"),
        'risefall': Quantity(value=t_final / 6, min_val=t_final / 10, max_val=t_final / 2, unit="s")
    }

    params_dict = {}
    for param_name in params & set(def_params.keys()):
        params_dict[param_name] = def_params[param_name]

    assert len(params_dict) == len(params), 'Error: different number of params than required!'

    return params_dict


def get_carrier_opt_params(drivers: set, gate_name: str) -> list:
    carrier_opt_params = {'framechange', 'freq'}
    return [[(gate_name, driver, 'carrier', carr_param)] for driver in drivers for carr_param in carrier_opt_params]


def set_param_in_pmap(exp: Experiment, param_val, ind_to_opt: int):
    params = exp.pmap.get_parameters_scaled().numpy()
    params[ind_to_opt] = param_val
    set_point_in_pmap(exp, params)


# TODO - try without constraints but by just transforming the parameter?
def opt_param(exp: Experiment, ind_to_opt: int, qubit_inds: list) -> Tuple[str, Quantity, Quantity]:
    param_name = exp.pmap.opt_map[ind_to_opt][0]
    orig_val = deepcopy(exp.pmap.get_parameter_dict()[param_name])

    dims = exp.pmap.model.dims

    def opt_func(param_val: float) -> float:
        set_param_in_pmap(exp, param_val, ind_to_opt)

        exp.compute_propagators()

        return sparse_unitary_infid_set(exp.propagators, exp.pmap.instructions, qubit_inds, dims)

    opt_res = minimize_scalar(opt_func, bounds=(-1, 1))

    set_param_in_pmap(exp, opt_res.x, ind_to_opt)

    opt_val = deepcopy(exp.pmap.get_parameter_dict()[param_name])

    return param_name, orig_val, opt_val


def coordinate_descent_opt_exp(exp: Experiment, qubit_inds: list, n_params: float) -> np.ndarray:
    n_iters = 0
    while True:
        n_iters += 1
        ind_to_opt = np.random.randint(n_params)
        param_name, prev_qty, curr_qty = opt_param(exp, ind_to_opt, qubit_inds)

        exp.compute_propagators()
        fid = calc_exp_fid(exp, qubit_inds)

        print(f'F={fid:.3f}')
        print(f'{param_name}: {prev_qty}-> {curr_qty}')


if __name__ == '__main__':
    np.random.seed(0)

    t_final = 7e-9
    exp, rx90p = setup_rx90_exp(t_final=t_final)

    gate = rx90p
    gate_name = gate.get_key()

    exp.set_opt_gates([gate_name])
    gate_seq = [gate_name]

    opt_gates = [gate_name]

    gates_per_driver = {'d1': {'gaussian_nonorm': 1}}

    opt_map_params = get_carrier_opt_params(set(gates_per_driver.keys()), gate_name)
    params_to_exclude_from_opt = {'t_final', 'delta'}
    for driver, driver_envelopes in gates_per_driver.items():
        for env_name, n_envelopes in driver_envelopes.items():
            for n_envelope in range(n_envelopes):
                env_id_name = env_name + str(n_envelope)
                # add the envelope to the gate
                params_names = ENV_PARAMS[env_name]
                def_params = get_params_dict(params_names, t_final)
                env = Envelope(name=env_id_name, normalize_pulse=True, params=def_params,
                               shape=envelopes.envelopes[env_name])
                gate.add_component(env, driver)

                # add opt params to opt map
                opt_map_params += get_opt_params_conf(driver, gate_name, env_id_name,
                                                      params_names - params_to_exclude_from_opt)

    exp.pmap.instructions = {gate_name: gate}
    exp.pmap.update_parameters()
    exp.pmap.set_opt_map(opt_map_params)

    coordinate_descent_opt_exp(exp, [0], n_params=len(opt_map_params))
