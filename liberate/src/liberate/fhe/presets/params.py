from collections import UserDict
from copy import deepcopy

data = {
    "bronze": {
        "logN": 14,
        "num_special_primes": 1,
        "devices": [0],
        "scale_bits": 40,
        "num_scales": None,
    },
    "silver": {
        "logN": 15,
        "num_special_primes": 2,
        "devices": [0],
        "scale_bits": 40,
        "num_scales": None,
    },
    "gold": {
        "logN": 16,
        "num_special_primes": 4,
        "devices": None,
        "scale_bits": 40,
        "num_scales": None,
    },
    "platinum": {
        "logN": 17,
        "num_special_primes": 6,
        "devices": None,
        "scale_bits": 40,
        "num_scales": None,
    },
    "bootstrapping": {
        "logN": 16,
        "num_special_primes": 4,
        "devices": [0],
        "scale_bits": 53,
        "num_scales": None,
        "bias_guard": False
    }
}


class Params(UserDict):
    def __getitem__(self, key):
        return deepcopy(self.data[key])


params = Params(data)
