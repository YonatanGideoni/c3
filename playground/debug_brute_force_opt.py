import os

from matplotlib import pyplot as plt

from c3.experiment import Experiment
from playground.brute_force_opt_gate import read_cached_opt_map_params, find_opt_env_for_gate, calc_exp_fid
from playground.plot_utils import plot_dynamics, get_init_state, wait_for_not_mouse_press, plot_signal, \
    plot_splitted_population


def run_gate_exp_opt(exp_path: str, exp_cache_dir, debug_cache_dir='debug_cache', gate_name='cnot[0, 1]'):
    exp = Experiment()
    exp.read_config(exp_path)
    opt_map_params = read_cached_opt_map_params(exp_cache_dir)

    gate = exp.pmap.instructions[gate_name]
    find_opt_env_for_gate(exp, gate, opt_map_params, cache_dir=debug_cache_dir, debug=True)


def plot_exps_in_dir(dir_path: str):
    for file_name in os.listdir(dir_path):
        if file_name.split('.')[-1] != 'hjson':
            continue

        exp_path = os.path.join(dir_path, file_name)
        exp = Experiment()
        exp.read_config(exp_path)
        exp.compute_propagators()

        fid = calc_exp_fid(exp)

        print()
        print(exp_path)
        print(f'Fid={fid:.3f}')

        init_state = get_init_state(exp)
        gate = list(exp.pmap.instructions.values())[0]
        gate_name = gate.get_key()
        plot_dynamics(exp, init_state, [gate_name], disp_legend=True)
        plt.title(f'{gate_name}, F={fid:.3f}')
        plt.pause(1e-12)

        plot_splitted_population(exp, init_state, [gate_name])
        plt.pause(1e-12)

        drivers_signals = gate.comps
        drivers_signals = {driver: {sig_name: sig for sig_name, sig in signals.items() if 'carrier' not in sig_name}
                           for driver, signals in drivers_signals.items()}
        awg = exp.pmap.generator.devices['AWG']
        t_final = gate.t_end
        plot_signal(awg, drivers_signals, t_final)

        wait_for_not_mouse_press(timeout=30)

        plt.close('all')


def plot_good_results(base_dir: str):
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        if subdir != 'good_infid_exps':
            plot_good_results(subdir_path)
        else:
            plot_exps_in_dir(subdir_path)


if __name__ == '__main__':
    # exp_path = 'autopt_cache/best_d2_gaussian_nonorm2.hjson'
    # exp_cache_dir = 'autopt_cache/d2_gaussian_nonorm2'
    #
    # run_gate_exp_opt(exp_path, exp_cache_dir)

    plot_good_results('cy_brute_force_cache')
