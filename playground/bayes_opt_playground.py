import os
import tempfile

import numpy as np
import pandas as pd
from modAL.acquisition import max_EI
from modAL.models import BayesianOptimizer
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


def calc_exp_fid(exp: Experiment, index: list) -> float:
    dims = exp.pmap.model.dims
    return (1 - unitary_infid_set(exp.propagators, exp.pmap.instructions, index, dims)).numpy()


def setup_experiment_opt_ctrl(exp: Experiment, maxiter: int = 15) -> OptimalControl:
    log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

    opt = OptimalControl(
        dir_path=log_dir,
        fid_func=sparse_unitary_infid_set,
        fid_subspace=["Q1"],  # TODO-set this automatically
        pmap=exp.pmap,
        algorithm=lbfgs,
        options={'maxiter': maxiter},
    )

    opt.set_exp(exp)

    return opt


def set_point_in_pmap(exp: Experiment, point: np.ndarray):
    exp.pmap.set_parameters_scaled(inv_transform_exp_params_input(point))


def transform_exp_params_input(x: np.ndarray) -> np.ndarray:
    return np.log((1 - x) / (1 + x))


def inv_transform_exp_params_input(z: np.ndarray) -> np.ndarray:
    return 2 / (1 + np.exp(-z)) - 1


def transform_fid_func_output(y: np.ndarray, MAX_VAL: float = 0.995) -> np.ndarray:
    y = np.clip(y, a_min=-np.inf, a_max=MAX_VAL)
    return np.log((1 - y) / y)


def inv_transform_fid_func_output(y: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-y))


def bayes_opt_exp(exp: Experiment, qubit_inds: list, max_sampled_points: int = 100,
                  INIT_TYPICAL_LENGTH: float = 0.1, POINTS_POOL_SIZE: int = 10 ** 6,
                  POINTS_POOL_PER_ITER: int = 10 ** 4) -> tuple:
    n_params = len(exp.pmap.get_opt_map())
    dims = exp.pmap.model.dims

    kernel = RBF(np.ones(n_params) * INIT_TYPICAL_LENGTH)
    gpr = GaussianProcessRegressor(kernel, random_state=0)

    # setup fidelity function
    fid_func = lambda experiment: sparse_unitary_infid_set(experiment.propagators, experiment.pmap.instructions,
                                                           qubit_inds, dims)

    sampled_points_data = []
    optimizer = BayesianOptimizer(estimator=gpr, query_strategy=max_EI)
    large_points_pool = transform_exp_params_input(2 * np.random.rand(POINTS_POOL_SIZE, n_params) - 1)
    np_rng = np.random.default_rng(seed=0)
    while len(sampled_points_data) < max_sampled_points:
        # find next point to sample via EI
        subpool = np_rng.choice(large_points_pool, size=POINTS_POOL_PER_ITER, replace=False)
        points_inds, opt_points = optimizer.query(subpool)
        opt_point = opt_points[0]

        # update posterior
        set_point_in_pmap(exp, opt_point)
        exp.compute_propagators()
        func_res = transform_fid_func_output(fid_func(exp))
        fid = calc_exp_fid(exp, qubit_inds)
        sampled_points_data.append({'obj': func_res, 'fid': fid, 'point': opt_point})

        optimizer.teach(opt_point.reshape(1, -1), np.array([func_res]))

        # TODO - get another point by optimizing the previous one via LBFGS

    return gpr, pd.DataFrame.from_records(sampled_points_data)


if __name__ == '__main__':
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

    t_final = 7e-9  # Time for single qubit gates
    sideband = 50e6

    # gaussian params
    gauss_params = {
        "amp": Quantity(value=0.36, min_val=0.0, max_val=5, unit="V"),
        "t_final": Quantity(
            value=t_final, min_val=0.5 * t_final, max_val=1.5 * t_final, unit="s"
        ),
        "xy_angle": Quantity(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
        "freq_offset": Quantity(
            value=-sideband - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit="Hz 2pi"
        ),
        "delta": Quantity(value=-1, min_val=-5, max_val=3, unit=""),
        "sigma": Quantity(value=t_final / 4, min_val=t_final / 8, max_val=t_final / 2, unit="s"),
    }

    gauss_env = pulse.Envelope(
        name="gauss",
        desc="Gaussian comp for single-qubit gates",
        params=gauss_params,
        shape=envelopes.gaussian_nonorm,
    )

    lo_freq_q1 = 5e9 + sideband
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
    rx90p_q1.add_component(gauss_env, "d1")

    single_q_gates = [rx90p_q1]
    gate = rx90p_q1
    gate_name = gate.get_key()

    parameter_map = ParameterMap(instructions=[gate], model=model, generator=generator)
    exp = Experiment(pmap=parameter_map)

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

    gpr, sampled_points = bayes_opt_exp(exp, qubit_inds=[0])
    opt_point = sampled_points.point.iloc[sampled_points.obj.argmax()]

    set_point_in_pmap(exp, opt_point)
    exp.compute_propagators()

    print(f'F={calc_exp_fid(exp, [0]):.3f}')
    exp.pmap.print_parameters()
