import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np

from c3.c3objs import Quantity
from c3.experiment import Experiment
from c3.libraries.envelopes import gauss_white_noise, gaussian_sigma, fourier_sin, rect
from c3.signal.pulse import Envelope
from playground.brute_force_opt_gate import calc_exp_fid, SIDEBAND
from playground.plot_utils import wait_for_not_mouse_press, plot_signal, plot_splitted_population, plot_dynamics, \
    get_init_state

if __name__ == '__main__':
    debug_plot = False

    # high_fid_exp_path = 'doubly_resonant_cx\d1_blackman_window2\good_infid_exps\d2_hann1_3.hjson'
    # high_fid_exp_path = 'cy_brute_force_cache\d1_flattop_risefall2\good_infid_exps\d2_blackman_window1_2.hjson'
    # high_fid_exp_path = 'high_anharm_cnot_50ns_all_signals_ftgu\d2_hann2\good_infid_exps\d1_hann1_0.hjson'
    high_fid_exp_path = 'high_anharm_cnot_50ns_all_signals_ftgu\d1_hann2\good_infid_exps\d2_gaussian_nonorm1_5.hjson'

    exp = Experiment()
    exp.read_config(high_fid_exp_path)

    exp.compute_propagators()
    print(f'Init fid: {calc_exp_fid(exp)}')

    for qubit in exp.pmap.model.subsystems.values():
        qubit.hilbert_dim = 6
    exp.compute_propagators()
    print(f'Many dimensions fid: {calc_exp_fid(exp)}')

    gate = list(exp.pmap.instructions.values())[0]
    gate_name = gate.get_key()

    t_final = gate.t_end

    sig_amp = Quantity(value=0.0, min_val=0.0, max_val=1., unit="V")
    params = {'amp': sig_amp,
              't_final': Quantity(value=t_final, min_val=0.0 * t_final, max_val=2.5 * t_final, unit="s"),
              'delta': Quantity(value=-1, min_val=-5, max_val=3, unit=""),
              'xy_angle': Quantity(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit='rad'),
              'freq_offset': Quantity(value=-53e6, min_val=-70 * 1e6, max_val=52 * 1e6, unit='Hz 2pi'), }

    env = Envelope(name='const_sig', normalize_pulse=False, params=params, shape=rect)
    gate.add_component(env, 'd1')

    base_instructions = deepcopy(exp.pmap.instructions)
    exp.pmap.instructions.update({gate.get_key(): gate})
    exp.pmap.update_parameters()

    const_sig_amps = 10 ** np.linspace(-10, 0, 25)
    fids = []
    for amp in const_sig_amps:
        print()
        sig_amp.set_value(amp)

        exp.pmap.model.update_model()
        exp.pmap.update_parameters()

        exp.compute_propagators()

        fid = calc_exp_fid(exp)
        fids.append(fid)
        print(f'Fid for constant signal of amplitude {amp:.2e}[V]: {fid}')

        if debug_plot:
            init_state = get_init_state(exp, energy_level=0)
            gate = list(exp.pmap.instructions.values())[0]
            plot_dynamics(exp, init_state, [gate_name], disp_legend=True)
            plt.title(f'{gate_name}, F={fid:.3f}')

            wait_for_not_mouse_press()

            plt.close('all')

    plt.scatter(const_sig_amps, 1 - np.array(fids))
    plt.grid()
    plt.title('Infidelity for different added constant signals for high anharmonicity CX')
    plt.xlabel('Signal amplitude[V]')
    plt.ylabel('Infidelity')
    plt.semilogx()
    plt.semilogy()

    plt.show()
