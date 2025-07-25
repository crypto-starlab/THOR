import numpy as np
from ..ckks import CkksEngine

def evaluate_polynomial_stockmeyer(engine:CkksEngine, p:np.array, x):
    """
    @param p: polynomial coeffs with increasing order
    @param x: input ciphertext
    @param s: baby-step size for Stockmeyer Algorithm
    """
    degree = len(p) - 1
    
    # 1. Baby steps: X2, X3 
    x2 = engine.square(x)
    x3 = engine.auto_ct_ct_mult(x, x2)
    x_powers = [None, x, x2, x3]
    
    def evaluate_baby_poly(coeffs:np.array):
        if len(coeffs) == 1:
            return coeffs[0]
        else:
            result = engine.scalar_mult(coeffs[1], x_powers[1])
            if len(coeffs) == 2:
                return engine.add_scalar(result, coeffs[0])
            elif len(coeffs) == 3:
                result = engine.auto_cc_add(result, engine.scalar_mult(coeffs[2], x_powers[2]))
                return engine.add_scalar(result, coeffs[0])
            else:
                result = engine.auto_cc_add(result, engine.scalar_mult(coeffs[2], x_powers[2]))
                ax3 = engine.auto_ct_ct_mult(x_powers[2], engine.scalar_mult(coeffs[3], x))
                result = engine.auto_cc_add(result, ax3)
                return engine.add_scalar(result, coeffs[0])

    # 2. Giant steps: X4, X8, X16
    x4 = engine.square(x2)
    x8 = engine.square(x4)
    
    if degree >=16:
        x16 = engine.square(x8)  # x^16
    
    #3. Stockmeyer's Algorithm
    if len(p) > 16:
        subpolys = np.split(p, len(p)//4)
        baby_results = [evaluate_baby_poly(subpoly) for subpoly in subpolys]
        gs1 = []
        for i in range(0, len(baby_results), 2):
            if i + 1 < len(baby_results):
                gs1.append(engine.auto_cc_add(baby_results[i], engine.auto_ct_ct_mult(baby_results[i+1], x4)))
            else:
                gs1.append(baby_results[i])
        gs2 = []
        for i in range(0, len(gs1), 2):
            if i + 1 < len(gs1):
                gs2.append(engine.auto_cc_add(gs1[i], engine.auto_ct_ct_mult(gs1[i+1], x8)))
            else:
                gs2.append(gs1[i])
        result = engine.auto_cc_add(gs2[0], engine.auto_ct_ct_mult(gs2[1], x16))
    
    elif len(p) == 16:
        subpolys = np.split(p, 4)
        baby_results = [evaluate_baby_poly(subpoly) for subpoly in subpolys]
        gs1 = []
        for i in range(0, 4, 2):
            gs1.append(engine.auto_cc_add(baby_results[i], engine.auto_ct_ct_mult(baby_results[i+1], x4)))
        result = engine.auto_cc_add(gs1[0], engine.auto_ct_ct_mult(gs1[1], x8))
        
    return result