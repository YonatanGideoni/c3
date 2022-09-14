from c3.experiment import Experiment
from playground.brute_force_opt_gate import read_cached_opt_map_params, find_opt_env_for_gate


def run_gate_exp_opt(exp_path: str, exp_cache_dir, debug_cache_dir='debug_cache', gate_name='cnot[0, 1]'):
    exp = Experiment()
    exp.read_config(exp_path)
    opt_map_params = read_cached_opt_map_params(exp_cache_dir)

    gate = exp.pmap.instructions[gate_name]
    find_opt_env_for_gate(exp, gate, opt_map_params, cache_dir=debug_cache_dir, debug=True)


if __name__ == '__main__':
    exp_path = 'autopt_cache/best_d2_gaussian_nonorm2.hjson'
    exp_cache_dir = 'autopt_cache/d2_gaussian_nonorm2'

    run_gate_exp_opt(exp_path, exp_cache_dir)
