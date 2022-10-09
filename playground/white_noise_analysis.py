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
    high_fid_exp_path = 'cy_brute_force_cache\d1_flattop_risefall2\good_infid_exps\d2_blackman_window1_2.hjson'

    exp = Experiment()
    exp.read_config(high_fid_exp_path)

    # exp.pmap.generator.devices['LO'].resolution *= 2
    # exp.pmap.generator.devices['DigitalToAnalog'].resolution *= 2

    exp.compute_propagators()
    print(f'Init fid: {calc_exp_fid(exp)}')

    for qubit in exp.pmap.model.subsystems.values():
        qubit.hilbert_dim = 6
    exp.compute_propagators()
    print(f'Many dimensions fid: {calc_exp_fid(exp)}')

    gate = list(exp.pmap.instructions.values())[0]

    t_final = gate.t_end

    amp_r = Quantity(value=[0.0], min_val=0.0, max_val=1., unit="V")
    params = {'amps': amp_r,
              't_final': Quantity(value=t_final, min_val=0.0 * t_final, max_val=2.5 * t_final, unit="s"),
              'delta': Quantity(value=-1, min_val=-5, max_val=3, unit=""),
              'xy_angle': Quantity(value=0.0, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit='rad'),
              # 'freq_offset': Quantity(value=-64.164e6, min_val=-70 * 1e6, max_val=-52 * 1e6, unit='Hz 2pi'), }
              'freq_offset': Quantity(value=-52.252e6, min_val=-70 * 1e6, max_val=52 * 1e6, unit='Hz 2pi'), }

    env = Envelope(name='white_noise_r', normalize_pulse=False, params=params, shape=fourier_sin)
    gate.add_component(env, 'd1')

    amp_i = Quantity(value=0.0, min_val=0.0, max_val=1., unit="V")
    params = {'amp': amp_i,
              't_final': Quantity(value=t_final, min_val=0.0 * t_final, max_val=2.5 * t_final, unit="s"),
              'delta': Quantity(value=-1, min_val=-5, max_val=3, unit=""),
              'xy_angle': Quantity(value=np.pi / 2, min_val=-0.5 * np.pi, max_val=2.5 * np.pi, unit='rad'),
              # 'freq_offset': Quantity(value=-64.164e6, min_val=-70 * 1e6, max_val=-52 * 1e6, unit='Hz 2pi'), }
              'freq_offset': Quantity(value=-52.252e6, min_val=-70 * 1e6, max_val=-52 * 1e6, unit='Hz 2pi'), }

    env = Envelope(name='white_noise_i', normalize_pulse=False, params=params, shape=fourier_sin)
    gate.add_component(env, 'd1')

    base_instructions = deepcopy(exp.pmap.instructions)
    exp.pmap.instructions.update({gate.get_key(): gate})
    exp.pmap.update_parameters()

    calc_exp_fid(exp)
    amps = 2 ** np.linspace(-20, 0, 15)
    fids = []
    n_trials_avg_over = 30
    for amp_val in amps:
        print()
        amp_r.set_value(amp_val / 2 ** 0.5)
        amp_i.set_value(amp_val / 2 ** 0.5)

        tot_fid = 1
        for _ in range(n_trials_avg_over):
            exp.compute_propagators()

            tot_fid *= calc_exp_fid(exp)

        avg_fid = tot_fid ** (1 / n_trials_avg_over)
        fids.append(avg_fid)
        print(f'Fid for white noise of amplitude {amp_val:.2e}: {avg_fid}')

        if debug_plot:
            init_state = get_init_state(exp, energy_level=4)
            gate = list(exp.pmap.instructions.values())[0]
            gate_name = gate.get_key()
            plot_dynamics(exp, init_state, [gate_name], disp_legend=True)
            plt.title(f'{gate_name}, F={avg_fid:.3f}')
            plt.pause(1e-12)

            drivers_signals = gate.comps
            drivers_signals = {driver: {sig_name: sig for sig_name, sig in signals.items() if 'carrier' not in sig_name}
                               for driver, signals in drivers_signals.items()}
            awg = exp.pmap.generator.devices['AWG']
            t_final = gate.t_end
            plot_signal(awg, drivers_signals, t_final)

            wait_for_not_mouse_press()

            plt.close('all')

    plt.scatter(amps, 1 - np.array(fids))
    plt.grid()
    plt.title('Infidelity for different white noise for CY')
    plt.xlabel('Gaussian white noise amplitude[V]')
    plt.ylabel('Infidelity')
    plt.semilogx()
    plt.semilogy()

    plt.show()
