import os
from collections import namedtuple

import pandas as pd
from matplotlib import pyplot as plt

from c3.experiment import Experiment
from playground.utils import calc_exp_fid

ExpFid = namedtuple('ExpFid', ('exp', 'fid'))
DRAGPulses = namedtuple('DRAGPulses', ('gaussian', 'gauss_der'))


def get_subdir_exps(dir_path: str) -> list:
    subdir_exps = []
    for file_name in os.listdir(dir_path):
        if file_name.split('.')[-1] != 'hjson':
            continue

        exp_path = os.path.join(dir_path, file_name)
        exp = Experiment()
        exp.read_config(exp_path)
        for qubit in exp.pmap.model.subsystems.values():
            qubit.hilbert_dim = 6  # to make sure no bouncing off the cutoff is going on
        exp.compute_propagators()

        fid = calc_exp_fid(exp)

        subdir_exps.append(ExpFid(exp, fid))

    return subdir_exps


def get_drag_results(base_dir: str) -> list:
    exps_fids = []
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        if subdir != 'good_infid_exps':
            exps_fids += get_subdir_exps(subdir_path)
        else:
            exps_fids += get_drag_results(subdir_path)
    return exps_fids


def get_pulses_from_exp(exp: Experiment) -> DRAGPulses:
    comps = exp.pmap.instructions['rx90p[0]'].comps['d1']

    if 'gaussian_nonorm2' in comps:
        return DRAGPulses(comps['gaussian_nonorm2'], comps['gaussian_der_mag_unity1'])
    return DRAGPulses(comps['gaussian_nonorm1'], comps['gaussian_der_mag_unity2'])


if __name__ == '__main__':
    drag_results_dir = 'drag_res'

    exp_per_anharm = {}
    for res_dir in os.listdir(drag_results_dir):
        MHz_ind = res_dir.find('MHz')
        anharm_start_ind = res_dir.find('_-') + 1
        anharm = int(res_dir[anharm_start_ind:MHz_ind])
        print(f'Finding results for anharm={anharm}MHz')

        exps_fids = get_drag_results(os.path.join(drag_results_dir, res_dir))

        if not exps_fids:
            continue

        exp_per_anharm[anharm] = max(exps_fids, key=lambda x: x.fid)

    pulses_per_anharm = {anharm: get_pulses_from_exp(exp_and_fid.exp) for anharm, exp_and_fid in exp_per_anharm.items()}
    fid_per_anharm = {anharm: exp_and_fid.fid for anharm, exp_and_fid in exp_per_anharm.items()}

    data = pd.DataFrame.from_records([{'anharm': anharm,
                                       'fid': fid_per_anharm[anharm],
                                       'gaussian': pulses.gaussian,
                                       'gauss_der': pulses.gauss_der}
                                      for anharm, pulses in pulses_per_anharm.items()])
    data['gauss_amp'] = data.gaussian.apply(lambda pulse: pulse.params['amp'])
    data['gauss_sigma'] = data.gaussian.apply(lambda pulse: pulse.params['sigma'])
    data['gauss_der_amp'] = data.gauss_der.apply(lambda pulse: pulse.params['amp'])
    data['gauss_der_sigma'] = data.gaussian.apply(lambda pulse: pulse.params['sigma'])
    data['rel_xy_angle'] = data.apply(lambda row: (row.gaussian.params['xy_angle'] -
                                                   row.gauss_der.params['xy_angle'])[0], axis='columns')

    data['amp_ratio'] = data.gauss_der_amp / data.gauss_amp
    # TODO - make sure to explicitly mention this absolute value (via the sign swap)
    data['exp_amp_ratio'] = -1e-6 / (data.anharm * data.gauss_sigma)  # 1e-6 because anharmonicity is measured in MHz

    # TODO - plot this such that the perfect fit is a straight line instead of a flat one?
    data['real_over_exp_amp_ratios'] = data.amp_ratio / data.exp_amp_ratio

    # TODO - throw the plots into functions
    data.plot.scatter(x='anharm', y='real_over_exp_amp_ratios', marker='X')
    plt.axhline(1., c='k', linestyle='dashed')

    fs = 14
    plt.xlabel('Anharmonicity [MHz]', fontsize=fs)
    plt.ylabel(r'$\frac{A_\mathrm{der}}{\tau|\Delta|A_\mathrm{pulse}}$', fontsize=fs + 6)

    plt.show()
