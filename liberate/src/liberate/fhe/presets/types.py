from collections import UserDict
from copy import deepcopy

data = {
    "sk": "secret key",
    "pk": "public key",
    "ksk": "key switch key",
    "rotk": "rotation key:",
    "galk": "galois key",
    "conjk": "conjugation key",
    "ct": "cipher text",
    "ctt": "cipher text triplet"

}


class Origins(UserDict):
    def __getitem__(self, key):
        return deepcopy(self.data[key])


origins = Origins(data)
