import copy

import numpy as np
import pandas as pd

from c3.c3objs import Quantity
from c3.experiment import Experiment
from c3.libraries import envelopes, algorithms
from c3.libraries.fidelities import unitary_infid_set
from c3.optimizers.optimalcontrol import OptimalControl
from c3.parametermap import ParameterMap
from c3.signal import pulse
from c3.signal.gates import Instruction
from playground.brute_force_opt_gate import SIDEBAND, get_2q_system
from playground.utils import calc_exp_fid


def get_freq_envs(n_freqs: int, t_final_2Q: float, max_freq: float = 300e6, max_amp: float = 3.,
                  max_init_amp: float = 5e-1):
    freqs_params = {
        'amps': Quantity(value=np.random.uniform(0, max_init_amp, n_freqs), min_val=0, max_val=max_amp, unit="V"),
        'phases': Quantity(value=np.random.uniform(-np.pi, np.pi, n_freqs), min_val=-np.pi, max_val=np.pi,
                           unit="rad"),
        'freqs': Quantity(value=np.random.uniform(0, max_freq, n_freqs), min_val=0, max_val=max_freq,
                          unit='Hz 2pi'),
        't_final': Quantity(value=t_final_2Q, min_val=0.5 * t_final_2Q, max_val=1.5 * t_final_2Q, unit="s"),
        'freq_offset': Quantity(value=-SIDEBAND - 3e6, min_val=-56 * 1e6, max_val=-52 * 1e6, unit='Hz 2pi'),
    }

    freq_envs_2Q_1 = pulse.Envelope(
        name="freqs1",
        params=freqs_params,
        shape=envelopes.fourier_sin
    )
    freq_envs_2Q_2 = pulse.Envelope(
        name="freqs2",
        params=copy.deepcopy(freqs_params),
        shape=envelopes.fourier_sin
    )

    return freq_envs_2Q_1, freq_envs_2Q_2


def setup_experiment_opt_ctrl(exp: Experiment, maxiter: int = 250) -> OptimalControl:
    n_qubits = len(exp.pmap.model.dims)
    fid_subspace = [f'Q{i + 1}' for i in range(n_qubits)]

    opt = OptimalControl(
        fid_func=unitary_infid_set,
        fid_subspace=fid_subspace,
        pmap=exp.pmap,
        algorithm=algorithms.lbfgs,
        options={'maxiter': maxiter},
    )

    opt.set_exp(exp)

    return opt


def benchmark_freq_basis(exp_builder: callable, t_final: float, max_n_freqs: int = 15,
                         n_trials: int = 10) -> pd.DataFrame:
    results = []
    for n_freqs in range(1, max_n_freqs + 1):
        print(f'Starting n_freqs={n_freqs}')
        for n_trial in range(n_trials):
            print(f'Run {n_trial + 1}/{n_trials}')

            exp, gate = exp_builder(t_final)
            gate_name = gate.get_key()

            freqs1, freqs2 = get_freq_envs(n_freqs, t_final)

            gate.add_component(freqs1, "d1")
            gate.add_component(freqs2, "d2")

            exp.pmap.instructions.update({gate_name: gate})
            exp.pmap.update_parameters()

            opt_gates = [gate_name]
            exp.set_opt_gates(opt_gates)

            gateset_opt_map = [
                [(gate_name, "d1", "freqs1", "amps")],
                [(gate_name, "d1", "freqs1", "freqs")],
                [(gate_name, "d1", "freqs1", "phases")],
                [(gate_name, "d1", "freqs1", "freq_offset")],
                [(gate_name, "d2", "freqs2", "amps")],
                [(gate_name, "d2", "freqs2", "freqs")],
                [(gate_name, "d2", "freqs2", "phases")],
                [(gate_name, "d2", "freqs2", "freq_offset")],
                [(gate_name, "d1", "carrier", "framechange")],
                [(gate_name, "d2", "carrier", "framechange")],
            ]
            exp.pmap.set_opt_map(gateset_opt_map)

            opt = setup_experiment_opt_ctrl(exp)
            opt.optimize_controls()

            exp.compute_propagators()
            results.append({'n_freqs': n_freqs, 'fid': calc_exp_fid(exp), 't_final': t_final})

    return pd.DataFrame.from_records(results)


def main():
    t_final = 45e-9
    qubit_lvls = 3

    def exp_builder(t_final):
        gate, model, generator = get_2q_system('cx', qubit_lvls, t_final)[:-1]
        parameter_map = ParameterMap(instructions=[gate], model=model, generator=generator)
        exp = Experiment(pmap=parameter_map)

        return exp, gate

    res = benchmark_freq_basis(exp_builder, t_final)
    res.to_csv('freq_basis_benchmark.csv')


if __name__ == '__main__':
    main()
