import matplotlib.pyplot as plt
from copy import deepcopy

import numpy as np

from c3.c3objs import Quantity
from c3.experiment import Experiment
from c3.libraries.envelopes import gauss_white_noise, gaussian_sigma, fourier_sin
from c3.signal.pulse import Envelope
from playground.brute_force_opt_gate import calc_exp_fid, SIDEBAND
from playground.plot_utils import wait_for_not_mouse_press, plot_signal, plot_splitted_population, plot_dynamics, \
    get_init_state

if __name__ == '__main__':
    debug_plot = False

    # high_fid_exp_path = 'doubly_resonant_cx\d1_blackman_window2\good_infid_exps\d2_hann1_3.hjson'
    # high_fid_exp_path = 'cy_brute_force_cache\d1_flattop_risefall2\good_infid_exps\d2_blackman_window1_2.hjson'
    high_fid_exp_path = 'high_anharm_cnot_50ns_all_signals_ftgu\d2_hann2\good_infid_exps\d1_hann1_0.hjson'

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

    freq_qty = exp.pmap.model.subsystems['Q1'].params['freq']
    base_freq = freq_qty.numpy()
    freqs_changes = 10 ** np.linspace(0, 10, 50)
    fids = []
    for delta_freq in freqs_changes:
        print()
        freq_qty.set_value(base_freq + delta_freq, extend_bounds=True)
        exp.pmap.model.update_model()
        exp.pmap.update_parameters()

        exp.compute_propagators()

        fid = calc_exp_fid(exp)
        fids.append(fid)
        print(f'Fid for qubit frequency offset of magnitude {delta_freq:.2e}[Hz]: {fid}')

        if debug_plot:
            init_state = get_init_state(exp, energy_level=0)
            gate = list(exp.pmap.instructions.values())[0]
            plot_dynamics(exp, init_state, [gate_name], disp_legend=True)
            plt.title(f'{gate_name}, F={fid:.3f}')

            wait_for_not_mouse_press()

            plt.close('all')

    plt.scatter(freqs_changes, 1 - np.array(fids))
    plt.grid()
    plt.title('Infidelity for different control qubit frequencies for high anharmonicity CX')
    plt.xlabel('Frequency offset[Hz]')
    plt.ylabel('Infidelity')
    plt.semilogx()
    plt.semilogy()

    plt.show()
