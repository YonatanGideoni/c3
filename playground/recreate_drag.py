import copy
import os

import numpy as np

from c3.c3objs import Quantity
from c3.experiment import Experiment
from c3.generator import devices
from c3.generator.generator import Generator
from c3.libraries import envelopes, tasks, chip, hamiltonians
from c3.model import Model
from c3.parametermap import ParameterMap
from c3.signal import pulse, gates
from playground.brute_force_opt_gate import LatentGridSamplingOptimiser, OptimiserParams


def get_1q_system(gate_name, __t_final, __anharm):
    lindblad = False
    dressed = True
    qubit_lvls = 3
    freq = 5e9
    init_temp = 0
    qubit_temp = 0
    sim_res = 100e9
    awg_res = 2e9
    sideband = 50e6
    lo_freq = 5e9 + sideband

    # ### MAKE MODEL
    q1 = chip.Qubit(
        name="Q1",
        desc="Qubit 1",
        freq=Quantity(
            value=freq,
            min_val=4.995e9,
            max_val=5.005e9,
            unit="Hz 2pi",
        ),
        anhar=Quantity(
            value=__anharm,
            min_val=-1e9,
            max_val=-1e6,
            unit="Hz 2pi",
        ),
        hilbert_dim=qubit_lvls,
        temp=Quantity(value=qubit_temp, min_val=0.0, max_val=0.12, unit="K"),
    )

    drive = chip.Drive(
        name="d1",
        desc="Drive 1",
        comment="Drive line 1 on qubit 1",
        connected=["Q1"],
        hamiltonian_func=hamiltonians.x_drive,
    )
    phys_components = [q1]
    line_components = [drive]

    model = Model(phys_components, line_components)
    model.set_lindbladian(lindblad)
    model.set_dressed(dressed)

    # ### MAKE GENERATOR
    generator = Generator(
        devices={
            "LO": devices.LO(name="lo", resolution=sim_res, outputs=1),
            "AWG": devices.AWG(name="awg", resolution=awg_res, outputs=1),
            "DigitalToAnalog": devices.DigitalToAnalog(
                name="dac", resolution=sim_res, inputs=1, outputs=1
            ),
            "Response": devices.Response(
                name="resp",
                rise_time=Quantity(value=0.3e-9, min_val=0.05e-9, max_val=0.6e-9, unit="s"),
                resolution=sim_res,
                inputs=1,
                outputs=1,
            ),
            "Mixer": devices.Mixer(name="mixer", inputs=2, outputs=1),
            "VoltsToHertz": devices.VoltsToHertz(
                name="v_to_hz",
                V_to_Hz=Quantity(value=1e9, min_val=0.9e9, max_val=1.1e9, unit="Hz/V"),
                inputs=1,
                outputs=1,
            ),
        },
        chains={
            "d1": {
                "LO": [],
                "AWG": [],
                "DigitalToAnalog": ["AWG"],
                "Response": ["DigitalToAnalog"],
                "Mixer": ["LO", "Response"],
                "VoltsToHertz": ["Mixer"],
            }
        },
    )

    # ### MAKE GATESET
    carrier_parameters = {
        "freq": Quantity(
            value=lo_freq,
            min_val=4.5e9,
            max_val=6e9,
            unit="Hz 2pi",
        ),
        "framechange": Quantity(value=0.0, min_val=-np.pi, max_val=3 * np.pi, unit="rad"),
    }
    carr = pulse.Carrier(
        name="carrier",
        desc="Frequency of the local oscillator",
        params=carrier_parameters,
    )

    rx90p = gates.Instruction(
        name="rx90p", t_start=0.0, t_end=__t_final, channels=["d1"], targets=[0]
    )

    rx90p.add_component(carr, "d1")

    if gate_name == 'rx90p':
        gate = rx90p
    else:
        raise NotImplementedError('Still need to implement option of creating exp with that gate!')

    return gate, model, generator


# TODO - set basis pulses
# TODO - make sure it works
if __name__ == '__main__':
    t_final = 7e-9
    min_anharm = -10e6
    max_anharm = -500e6

    for anharm in np.linspace(min_anharm, max_anharm, num=15):
        gate, model, generator = get_1q_system('rx90p', __t_final=t_final, __anharm=anharm)
        dir = f'rx90_{t_final * 1e9:.0f}ns_{anharm / 1e6:.0f}MHz_drag_recr_trial'

        if not os.path.isdir(dir):
            os.mkdir(dir)

        parameter_map = ParameterMap(instructions=[gate], model=model, generator=generator)
        exp = Experiment(pmap=parameter_map)

        opt_params = OptimiserParams(normalise_pulses=False, max_amp=2, min_amp=1e-2,
                                     rel_envs=('gaussian_nonorm', 'gaussian_der_nonorm'))

        LatentGridSamplingOptimiser(optimiser_params=opt_params, verbose=True, debug=True) \
            .optimize_gate(exp, gate, cache_dir=dir, n_pulses_to_add=2, opt_all_at_once=False)
