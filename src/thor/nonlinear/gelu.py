import numpy as np
from liberate.fhe.data_struct import DataStruct

from ..ckks import CkksEngine
from .polynomial import evaluate_polynomial_stockmeyer

def he_gelu(engine:CkksEngine, x:np.ndarray, sk=None, initial_btp=False):
    """
    Input is 2*x
    Output is gelu(x)/2 inorder to merge bootstrap at the next step
    """
    # print("gelu input level: ", x[0,0].level)
    gelu_x = np.full_like(x, fill_value=None, dtype=DataStruct)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]//2):
            x0 = x[i,j]
            x1 = x[i,j+x.shape[1]//2]
            tanh_x0 = he_tanh_single(engine, x0)
            tanh_x1 = he_tanh_single(engine, x1)
            one_plus_tanhx0 = engine.add_scalar(tanh_x0, 1/2) #Level: 20
            one_plus_tanhx1 = engine.add_scalar(tanh_x1, 1/2) #Level: 20
            x0_scaled = engine.mult_int_scalar(x0, 64)
            x1_scaled = engine.mult_int_scalar(x1, 64)
            gelu_x[i, j] = engine.auto_ct_ct_mult(x0_scaled, one_plus_tanhx0) #Level: 21
            gelu_x[i, j+x.shape[1]//2] = engine.auto_ct_ct_mult(x1_scaled, one_plus_tanhx1) #Level: 21
    return gelu_x

def he_tanh_single(engine:CkksEngine, x:DataStruct):
    """
    @param x: input ciphertext, scaled by 1/64
    """
    p1 = [-1.06240033e-05,  1.64454894e-04, -5.83533517e-04, -3.80912692e-04,
         2.24431193e-03,  8.92295204e-03, -1.05277477e-02, -1.91827040e-02,
        -2.04634786e-01,  4.54014410e-01, -5.40759203e-01,  5.67745523e+00,
        -1.36433727e+01,  1.82574621e+01, -8.48849601e+01,  1.28686741e+02,
         3.66720281e+02, -1.01400159e+03, -1.26278856e+02,  2.21728878e+03,
        -9.95421415e+02, -2.31059465e+03,  1.73583957e+03,  1.27394360e+03,
        -1.27836230e+03, -3.66781716e+02,  4.79663919e+02,  4.94610178e+01,
        -9.06754761e+01, -2.36515790e+00,  8.74311855e+00,  1.62838703e-02]

    p2 = [-1.70270667e+02,  6.81076279e+01,  1.79197364e+03, -6.81621043e+02,
        -8.49256169e+03,  3.05629446e+03,  2.39579397e+04, -8.10435126e+03,
        -4.48145152e+04,  1.41297616e+04,  5.86197512e+04, -1.70371505e+04,
        -5.51326382e+04,  1.45532495e+04,  3.77866438e+04, -8.87673890e+03,
        -1.89514802e+04,  3.84972853e+03,  6.94169727e+03, -1.16901058e+03,
        -1.84658407e+03,  2.41693754e+02,  3.54452276e+02, -3.24499570e+01,
        -4.91918227e+01,  2.58122977e+00,  5.78392852e+00, -9.45171527e-02]
    p1.reverse()
    p2.reverse()
    p1 = np.array(p1) 
    p2 = np.array(p2) * 0.5
    p1_x = evaluate_polynomial_stockmeyer(engine, p1, x)
    tanhx = evaluate_polynomial_stockmeyer(engine, p2, p1_x)

    return tanhx
