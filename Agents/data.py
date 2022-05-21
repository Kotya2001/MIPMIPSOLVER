import numpy as np
import random
from typing import NamedTuple


class Data(NamedTuple):
    coefs: np.ndarray
    constraits: np.ndarray
    bounds: np.ndarray
    ubound: int
    m: int
    n: int
    steps_per_forward: int
    r_lenght: int


def generateData(m: int, n: int, ubound: int, seed: int, steps_per_forward: int, r_lenght: int) -> NamedTuple:
    rand = np.random.RandomState(seed)
    coefs = np.round(rand.random_sample(m) * 5, 1)
    constraits = np.round(rand.random_sample((n, m)) * 10 * (rand.random_sample(m) * (coefs / 5) * 0.3 + 1), 1)
    bounds = np.round(constraits.sum(axis=1) * (rand.random_sample(n) * 0.5 + 0.3), 0)
    data = Data(coefs=coefs, constraits=constraits, bounds=bounds, ubound=ubound, m=m, n=n, r_lenght=r_lenght,
                steps_per_forward=steps_per_forward)
    return data


columns, strings, upperbound, s, steps, av_lenght = 10, 7, 3, 4, 11, 5
mip_task = generateData(columns, strings, upperbound, s, steps, av_lenght)
