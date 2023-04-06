from c3.experiment import Experiment
from c3.libraries.fidelities import unitary_infid_set


def calc_exp_fid(exp: Experiment) -> float:
    dims = exp.pmap.model.dims
    index = list(range(len(dims)))
    return (1 - unitary_infid_set(exp.propagators, exp.pmap.instructions, index, dims)).numpy()
