import sys
import os

conf_path = os.getcwd()
sys.path.append(conf_path)

import numpy as np

from c3.experiment import Experiment
from c3.parametermap import ParameterMap
from playground.brute_force_opt_gate import optimize_gate, get_2q_system


def run_cx(t_final: float, base_dir: str = 'cx_qsl'):
    gate, model, generator = get_2q_system('cx', __t_final=t_final)
    dir = os.path.join(f'{base_dir}', f'{t_final * 1e9:.0f}ns_ftgu')

    if not os.path.isdir(dir):
        os.makedirs(dir)

    parameter_map = ParameterMap(instructions=[gate], model=model, generator=generator)
    exp = Experiment(pmap=parameter_map)

    optimize_gate(exp, gate, cache_dir=dir, n_pulses_to_add=2, opt_all_at_once=True, debug=False)


if __name__ == '__main__':
    for t_final in np.arange(10e-9, 41e-9, 5e-9):
        print(f'Optimising CX for t_final={t_final * 10 ** 9:.0f}[ns]')
        run_cx(t_final, base_dir='cx_qsl_initial_ftgu_2pulse')
