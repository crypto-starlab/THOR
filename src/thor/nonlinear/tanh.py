import numpy as np
from liberate.fhe.data_struct import DataStruct

from ..ckks import CkksEngine
from .polynomial import evaluate_polynomial_stockmeyer


def he_tanh(engine:CkksEngine, x, min_x=-40, max_x=40, scale=40):
    """
    @params: x is scaled by 1/40, level:19
    """
    tanh_x = np.full_like(x, fill_value=None, dtype=DataStruct)
    for i in range(x.shape[0]):
        tanh_x[i] = he_tanh_single(engine, x[i], min_x, max_x, scale)
    return tanh_x

def he_tanh_single(engine:CkksEngine, normed_x, min_x=-40, max_x=40, scale=40): 
    """
    @params: normed_x is x scaled by 1/40, level:19
    """
    p1 = [-7.14529052e+03, -7.76519925e+01,  2.74279201e+04,  2.45150249e+02,
            -4.25793697e+04, -3.01953016e+02,  3.42189880e+04,  1.82989351e+02,
            -1.51158283e+04, -5.64098990e+01,  3.58757327e+03,  8.17596753e+00,
            -4.13341496e+02, -4.29024545e-01,  1.95056729e+01,  2.06201784e-03]
    p2 = [-9.02573450e-03, -1.12320034e-04,  1.08762008e-01,  7.96793166e-04,
            -5.41327356e-01, -1.42873183e-03,  1.46476749e+00, -2.22416152e-03,
            -2.43259032e+00,  1.17381072e-02,  2.74974898e+00, -1.77631073e-02,
            -2.38934873e+00,  1.30194294e-02,  2.02874846e+00, -4.08442578e-03]
    p1.reverse()
    p2.reverse()
    p1 = np.array(p1)
    p2 = np.array(p2)
    #Level 21 -> 15 -> 23 -> 15
    normed_x = engine.bootstrap(normed_x)
    tanh_x = evaluate_polynomial_stockmeyer(engine, p1, normed_x)
    tanh_x = evaluate_polynomial_stockmeyer(engine, p2, tanh_x)
    tanh_x = engine.bootstrap(tanh_x)
    return tanh_x

