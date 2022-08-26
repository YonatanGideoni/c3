import os
import tempfile
from typing import Callable

from c3.experiment import Experiment
import c3.libraries.algorithms as algorithms
import c3.libraries.fidelities as fidelities
from c3.optimizers.optimalcontrol import OptimalControl
from playground.plot_utils import wait_for_not_mouse_press, plot_signal


def setup_experiment_opt_ctrl(exp: Experiment, fid_func: Callable, maxiter: int = 5) -> OptimalControl:
    log_dir = os.path.join(tempfile.TemporaryDirectory().name, "c3logs")

    opt = OptimalControl(
        dir_path=log_dir,
        fid_func=fid_func,
        fid_subspace=["Q1", "Q2"],  # TODO-set this automatically
        pmap=exp.pmap,
        algorithm=algorithms.lbfgs,
        options={'maxiter': maxiter},
        run_name='solution_relaxation'
    )

    opt.set_exp(exp)

    return opt


def calc_exp_fid(exp: Experiment, index: list) -> float:
    dims = exp.pmap.model.dims
    return 1 - fidelities.unitary_infid_set(exp.propagators, exp.pmap.instructions, index, dims)


GATESET_OPT_MAP = [['cnot[0, 1]-d1-carrier-framechange'],
                   ['cnot[0, 1]-d2-carrier-framechange'],
                   ['cnot[0, 1]-d1-gauss0-sigma'],
                   ['cnot[0, 1]-d1-gauss0-freq_offset'],
                   ['cnot[0, 1]-d1-gauss0-amp'],
                   ['cnot[0, 1]-d1-gauss0-xy_angle'],
                   ['cnot[0, 1]-d1-gauss1-sigma'],
                   ['cnot[0, 1]-d1-gauss1-freq_offset'],
                   ['cnot[0, 1]-d1-gauss1-amp'],
                   ['cnot[0, 1]-d1-gauss1-xy_angle'],
                   ['cnot[0, 1]-d1-gauss2-sigma'],
                   ['cnot[0, 1]-d1-gauss2-freq_offset'],
                   ['cnot[0, 1]-d1-gauss2-amp'],
                   ['cnot[0, 1]-d1-gauss2-xy_angle'],
                   ['cnot[0, 1]-d1-gauss3-sigma'],
                   ['cnot[0, 1]-d1-gauss3-freq_offset'],
                   ['cnot[0, 1]-d1-gauss3-amp'],
                   ['cnot[0, 1]-d1-gauss3-xy_angle'],
                   ['cnot[0, 1]-d1-gauss4-sigma'],
                   ['cnot[0, 1]-d1-gauss4-freq_offset'],
                   ['cnot[0, 1]-d1-gauss4-amp'],
                   ['cnot[0, 1]-d1-gauss4-xy_angle'],
                   ['cnot[0, 1]-d1-gauss5-sigma'],
                   ['cnot[0, 1]-d1-gauss5-freq_offset'],
                   ['cnot[0, 1]-d1-gauss5-amp'],
                   ['cnot[0, 1]-d1-gauss5-xy_angle'],
                   ['cnot[0, 1]-d1-gauss6-sigma'],
                   ['cnot[0, 1]-d1-gauss6-freq_offset'],
                   ['cnot[0, 1]-d1-gauss6-amp'],
                   ['cnot[0, 1]-d1-gauss6-xy_angle'],
                   ['cnot[0, 1]-d1-gauss7-sigma'],
                   ['cnot[0, 1]-d1-gauss7-freq_offset'],
                   ['cnot[0, 1]-d1-gauss7-amp'],
                   ['cnot[0, 1]-d1-gauss7-xy_angle'],
                   ['cnot[0, 1]-d1-gauss8-sigma'],
                   ['cnot[0, 1]-d1-gauss8-freq_offset'],
                   ['cnot[0, 1]-d1-gauss8-amp'],
                   ['cnot[0, 1]-d1-gauss8-xy_angle'],
                   ['cnot[0, 1]-d1-gauss9-sigma'],
                   ['cnot[0, 1]-d1-gauss9-freq_offset'],
                   ['cnot[0, 1]-d1-gauss9-amp'],
                   ['cnot[0, 1]-d1-gauss9-xy_angle'],
                   ['cnot[0, 1]-d2-gauss0-sigma'],
                   ['cnot[0, 1]-d2-gauss0-freq_offset'],
                   ['cnot[0, 1]-d2-gauss0-amp'],
                   ['cnot[0, 1]-d2-gauss0-xy_angle'],
                   ['cnot[0, 1]-d2-gauss1-sigma'],
                   ['cnot[0, 1]-d2-gauss1-freq_offset'],
                   ['cnot[0, 1]-d2-gauss1-amp'],
                   ['cnot[0, 1]-d2-gauss1-xy_angle'],
                   ['cnot[0, 1]-d2-gauss2-sigma'],
                   ['cnot[0, 1]-d2-gauss2-freq_offset'],
                   ['cnot[0, 1]-d2-gauss2-amp'],
                   ['cnot[0, 1]-d2-gauss2-xy_angle'],
                   ['cnot[0, 1]-d2-gauss3-sigma'],
                   ['cnot[0, 1]-d2-gauss3-freq_offset'],
                   ['cnot[0, 1]-d2-gauss3-amp'],
                   ['cnot[0, 1]-d2-gauss3-xy_angle'],
                   ['cnot[0, 1]-d2-gauss4-sigma'],
                   ['cnot[0, 1]-d2-gauss4-freq_offset'],
                   ['cnot[0, 1]-d2-gauss4-amp'],
                   ['cnot[0, 1]-d2-gauss4-xy_angle'],
                   ['cnot[0, 1]-d2-gauss5-sigma'],
                   ['cnot[0, 1]-d2-gauss5-freq_offset'],
                   ['cnot[0, 1]-d2-gauss5-amp'],
                   ['cnot[0, 1]-d2-gauss5-xy_angle'],
                   ['cnot[0, 1]-d2-gauss6-sigma'],
                   ['cnot[0, 1]-d2-gauss6-freq_offset'],
                   ['cnot[0, 1]-d2-gauss6-amp'],
                   ['cnot[0, 1]-d2-gauss6-xy_angle'],
                   ['cnot[0, 1]-d2-gauss7-sigma'],
                   ['cnot[0, 1]-d2-gauss7-freq_offset'],
                   ['cnot[0, 1]-d2-gauss7-amp'],
                   ['cnot[0, 1]-d2-gauss7-xy_angle'],
                   ['cnot[0, 1]-d2-gauss8-sigma'],
                   ['cnot[0, 1]-d2-gauss8-freq_offset'],
                   ['cnot[0, 1]-d2-gauss8-amp'],
                   ['cnot[0, 1]-d2-gauss8-xy_angle'],
                   ['cnot[0, 1]-d2-gauss9-sigma'],
                   ['cnot[0, 1]-d2-gauss9-freq_offset'],
                   ['cnot[0, 1]-d2-gauss9-amp'],
                   ['cnot[0, 1]-d2-gauss9-xy_angle']]
if __name__ == '__main__':
    cfg_path = 'two_qubits_entanglement_gauss_raw_var_sigma.hjson'
    exp = Experiment()
    exp.read_config(filepath=cfg_path)
    exp.pmap.set_opt_map(GATESET_OPT_MAP)
    awg = exp.pmap.generator.devices['AWG']

    drivers_signals = exp.pmap.instructions['cnot[0, 1]'].comps
    drivers_signals = {driver: {sig_name: sig for sig_name, sig in signals.items() if 'carrier' not in sig_name}
                       for driver, signals in drivers_signals.items()}

    reg_strength = 1e-10
    reg_fctr = 2
    while True:
        fid_func = lambda *args, **kwargs: fidelities.sparse_unitary_infid_set(*args,
                                                                               reg_strength=reg_strength,
                                                                               loss_func_type='sumOverSqrtMax',
                                                                               **kwargs)
        opt = setup_experiment_opt_ctrl(exp, fid_func)
        opt.optimize_controls()

        exp.compute_propagators()
        fid = calc_exp_fid(exp, [0, 1])

        print(f'Fidelity:      {fid:.3f}')
        print(f'Reg. strength: {reg_strength:.2e}')

        plot_signal(awg, drivers_signals, t_final=45e-9)
        wait_for_not_mouse_press()

        reg_strength *= reg_fctr
