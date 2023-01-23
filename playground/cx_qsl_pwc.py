import sys
import os

from matplotlib import pyplot as plt

conf_path = os.getcwd()
sys.path.append(conf_path)

import numpy as np

from c3.c3objs import Quantity
from c3.libraries.envelopes import pwc
from c3.signal.gates import Instruction
from c3.signal.pulse import Envelope
from playground.plot_utils import get_init_state, plot_dynamics, plot_splitted_population, plot_signal
from c3.experiment import Experiment
from c3.parametermap import ParameterMap
from playground.brute_force_opt_gate import get_2q_system, calc_exp_fid, setup_experiment_opt_ctrl


def add_pwc_pulses(exp: Experiment, gate: Instruction, n_slices: int, max_amp: float):
    gen_slices = lambda: np.clip(max_amp / 2 * np.random.randn(n_slices), -max_amp, max_amp)

    pwc1_params = {
        "inphase": Quantity(value=gen_slices(), unit="V"),
        "quadrature": Quantity(value=gen_slices(), unit="V"),
        "amp": Quantity(value=1.0, unit="V"),
        "xy_angle": Quantity(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
        "freq_offset": Quantity(value=0, min_val=-5 * 1e6, max_val=5 * 1e6, unit="Hz 2pi"),
    }

    pwc2_params = {
        "inphase": Quantity(value=gen_slices(), unit="V"),
        "quadrature": Quantity(value=gen_slices(), unit="V"),
        "amp": Quantity(value=1.0, unit="V"),
        "xy_angle": Quantity(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit="rad"),
        "freq_offset": Quantity(value=0, min_val=-5 * 1e6, max_val=5 * 1e6, unit="Hz 2pi"),
    }

    pwc1 = Envelope(
        name="pwc1",
        desc="piecewise constant",
        params=pwc1_params,
        shape=pwc,
    )
    pwc2 = Envelope(
        name="pwc2",
        desc="piecewise constant",
        params=pwc2_params,
        shape=pwc,
    )

    gate.add_component(pwc1, "d1")
    gate.add_component(pwc2, "d2")
    gate_name = gate.get_key()

    gateset_opt_map = []
    for driver_num in range(1, 3):
        gateset_opt_map += [
            [
                (gate_name, f'd{driver_num}', f"pwc{driver_num}", "amp"),
            ],
            [
                (gate_name, f'd{driver_num}', f"pwc{driver_num}", "inphase"),
            ],
            [
                (gate_name, f'd{driver_num}', f"pwc{driver_num}", "quadrature"),
            ]
        ]

    exp.pmap.instructions = {gate_name: gate}
    exp.pmap.update_parameters()
    exp.pmap.set_opt_map(gateset_opt_map)


def run_cx(t_final: float, base_dir: str = 'cx_qsl', max_iter: int = 500):
    gate, model, generator = get_2q_system('cx', __t_final=t_final, qubit_lvls=6)
    dir = os.path.join(f'{base_dir}', f'{t_final * 1e9:.2f}ns_pwc')

    if not os.path.isdir(dir):
        os.makedirs(dir)

    parameter_map = ParameterMap(instructions=[gate], model=model, generator=generator)
    exp = Experiment(pmap=parameter_map)
    awg_res = exp.pmap.generator.devices['AWG'].resolution
    n_slices = int(t_final * awg_res)

    add_pwc_pulses(exp, gate, n_slices, MAX_AMP)

    exp_opt = setup_experiment_opt_ctrl(exp, maxiter=max_iter)
    exp_opt.optimize_controls()

    exp.compute_propagators()
    fid = calc_exp_fid(exp)
    exp.write_config(os.path.join(dir, 'exp.hjson'))

    init_state = get_init_state(exp, energy_level=4)
    gate = list(exp.pmap.instructions.values())[0]
    gate_name = gate.get_key()
    plot_dynamics(exp, init_state, [gate_name], disp_legend=True)
    plt.title(f'{gate_name}, F={fid:.3f}')
    plt.savefig(os.path.join(dir, 'dynamics.png'))
    plt.close()

    plot_splitted_population(exp, init_state, [gate_name])
    plt.savefig(os.path.join(dir, 'split_pop.png'))
    plt.close()

    drivers_signals = gate.comps
    drivers_signals = {driver: {sig_name: sig for sig_name, sig in signals.items() if 'carrier' not in sig_name}
                       for driver, signals in drivers_signals.items()}
    awg = exp.pmap.generator.devices['AWG']
    plot_signal(awg, drivers_signals, t_final, n_points=n_slices)
    plt.savefig(os.path.join(dir, 'signal.png'))
    plt.close()

    return fid


if __name__ == '__main__':
    MAX_AMP = 0.5
    min_fid = 0.99

    max_t_final = 100e-9
    min_t_final = 0
    fid_per_t_final = {}
    while max_t_final - min_t_final > 0.5e-9:
        t_final = 0.5 * (max_t_final + min_t_final)

        print(f'Optimising CX for t_final={t_final * 10 ** 9:.2f}[ns]')
        fid = run_cx(t_final, base_dir=f'cx_qsl_pwc_max_amp_{MAX_AMP}')
        fid_per_t_final[t_final] = fid

        if fid > min_fid:
            max_t_final = t_final
        else:
            min_t_final = t_final

    print(fid_per_t_final)
