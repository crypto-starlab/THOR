import numpy as np
from liberate.fhe.data_struct import DataStruct

from ..ckks import CkksEngine
from .polynomial import evaluate_polynomial_stockmeyer

def he_softmax1(engine:CkksEngine, x, attention_mask, rescale=False, debug=False, sk=None):
    return he_softmax(engine, x, attention_mask, rescale=rescale, min_x=-27.2493, max_x=21.72692, n=2, l=2, inv_epsilon=2**(-11),output_alpha=0.01, debug=False, sk=sk)
def he_softmax2(engine:CkksEngine, x, attention_mask, rescale=False, debug=False, sk=None):
    return he_softmax(engine, x, attention_mask, rescale=rescale, min_x=-70, max_x=70, n=2, l=4,inv_epsilon=2**(-18),output_alpha=0.01, debug=False, sk=sk)

def he_softmax(engine:CkksEngine, u, attention_mask, rescale, min_x, max_x, n, l, inv_epsilon, output_alpha, debug, sk):
    """
    @param engine: CkksEngine
    @param u: np.ndarray, ct of level 14
    @param attention_mask: np.ndarray
    @param: rescale: bool, if true, return the scaled result
    """
    if debug == True and sk == None:
        raise ValueError("sk must be provided for debug mode")
    #Compute exp(U)    
    if max_x<30:
        exp_u = [he_exp1(engine,ct,min_x=min_x, max_x=max_x, n=n) for ct in u]
    else:
        exp_u = [he_exp2(engine,ct,min_x=min_x, max_x=max_x, n=n) for ct in u]
    
    for i in range(8):
        exp_u[i] = engine.rescale(engine.pt_ct_mult(attention_mask[i],exp_u[i]))
    
    #Compute sigma(exp(U))
    sigma_exp = exp_u[0]
    for i in range(1, len(exp_u)):
        sigma_exp = engine.cc_add(sigma_exp, exp_u[i])
    sigma_exp = engine.rotsum(sigma_exp,interval=2**11)
    
    #Compute 1/sigma(exp(U))
    internal_alpha=0.1
    if l>1:
        masking = np.array(([1]*12+[0.0]*4)*2**11)
        enc_one = engine.encode_and_encrypt(masking, level=engine.num_levels-sigma_exp.level)
        inv_D,D_delta,output_precision = he_inv(engine, enc_one, sigma_exp, epsilon=inv_epsilon, alpha=internal_alpha/10)          
        
        # 0 iteration in the for loop
        for _ in range(int(np.log2(l)-1)):
            exp_u, inv_D, D_delta, output_precision = update_inv_D(engine, exp_u, attention_mask, inv_D, D_delta, output_precision, alpha=internal_alpha)
        exp_u, inv_D, D_delta, output_precision = update_inv_D(engine, exp_u, attention_mask, inv_D, D_delta, output_precision, alpha=output_alpha, final_inv=True)
    elif l==1:
        masking = np.array(([1]*12+[0.01]*4)*2**7)
        enc_one = engine.encode_and_encrypt(masking, level=engine.num_levels-sigma_exp.level)
        inv_D,D_delta,output_precision = he_inv(engine, enc_one, sigma_exp,epsilon=inv_epsilon,alpha=output_alpha)

    cplx_softmax1=[]
    cplx_softmax2=[]
    rotated_inv_D = [inv_D]
    for i in range(15):
        rotated_inv_D.append(engine.rotate_left(rotated_inv_D[-1],-2**11))
    
    for i in range(4):
        exp_cplx=engine.cc_add(exp_u[i],engine.imult(exp_u[i+4]))
        exp_cplx = engine.mult_int_scalar(exp_cplx, int(1/(2*D_delta))+1)        
        if exp_cplx.level < 4:
            exp_cplx = engine.bootstrap(exp_cplx)
            exp_cplx = engine.level_up(exp_cplx, 16)
        for j in range(16):
            masked_softmax= engine.auto_ct_ct_mult(exp_cplx, rotated_inv_D[j])
            if rescale:
                masked_softmax = engine.rescale(masked_softmax)
            softmax_copied=engine.rotsum(masked_softmax,interval=2**11)
            softmax_copied_real=engine.cc_add(softmax_copied,engine.conjugate(softmax_copied))
            cplx_softmax1+=[softmax_copied_real]
            softmax_copied_imag=engine.cc_sub(engine.conjugate(softmax_copied),softmax_copied)
            softmax_copied_imag=engine.imult(softmax_copied_imag)
            cplx_softmax2+=[softmax_copied_imag]
    softmax_128=cplx_softmax1+cplx_softmax2
    result = np.full((128,), None, dtype=object)
    for i in range(128):
        result[i] = softmax_128[i]
    return result
    
def update_inv_D(engine, exp_u, attention_mask, inv_D, D_delta, output_precision, alpha, final_inv=False):
    exp_2u = np.full((8,), None, dtype=object)

    # Rescale inv_D with attention_mask
    inv_D_list = [engine.rescale(engine.pt_ct_mult(attention_mask[i], inv_D)) for i in range(8)]

    # Adjust levels and perform computations
    for i in range(4):
        exp_u[i], inv_D_list[i] = engine.auto_level(exp_u[i], inv_D_list[i])
        exp_u[i + 4], inv_D_list[i + 4] = engine.auto_level(exp_u[i + 4], inv_D_list[i + 4])

        k = max(int(1 / D_delta / 2), 1)

        # Perform softmax calculation
        partial_softmax0 = engine.square(engine.mult_int_scalar(engine.ct_ct_mult(engine.cc_add(exp_u[i], exp_u[i]), inv_D_list[i]), k))
        partial_softmax1 = engine.square(engine.mult_int_scalar(engine.ct_ct_mult(engine.cc_add(exp_u[i + 4], exp_u[i + 4]), inv_D_list[i + 4]), k))

        exp_2u[i] = partial_softmax0
        exp_2u[i + 4] = partial_softmax1

    # Sum up the results
    summation = exp_2u[0]
    for i in range(1, len(exp_2u)):
        summation = engine.cc_add(summation, exp_2u[i])

    summation = engine.mult_int_scalar(summation,2**6)
    summation = engine.bootstrap(summation)
    summation = engine.mult_scalar(summation,1/2**6)
    summation = engine.rotsum(summation, interval=2**11)

    # Generate mask for final computation
    masking = np.array(([1] * 12 + [0.00] * 4) * 2**11)
    
    # Inverse calculation
    epsilon2 = output_precision / 128 / 2
    enc_one = engine.encode_and_encrypt(masking, level=engine.num_levels - summation.level)
    inv_D, D_delta, output_precision = he_inv(engine, enc_one, summation, epsilon=epsilon2, alpha=alpha / 10)
    if final_inv:
        masking = np.array(([1] * 12 + [0] * 4) * (2**7))
    inv_D = engine.cm_mult(inv_D, masking)
    return exp_2u, inv_D, D_delta, output_precision

def he_inv(engine:CkksEngine, numerator:DataStruct, denominator:DataStruct, epsilon:float, alpha:float, delta=1):
    d = 0
    an = Ciphertext(numerator, delta)
    bn = Ciphertext(denominator, delta)
    en = epsilon 
    while en<1-alpha:
        
        if an.ciphertext.level_calc > engine.num_levels - 3:
            scale_adjust=int(1/(bn.delta*4)/2)
            
            if scale_adjust>1:
                #Denoise imaginary part
                conj_an=Ciphertext(engine.conjugate(an.ciphertext), an.delta)
                an=Ciphertext(engine.add(an.ciphertext,conj_an.ciphertext), an.delta*2)
                conj_bn=Ciphertext(engine.conjugate(bn.ciphertext),bn.delta)
                bn=Ciphertext(engine.add(bn.ciphertext,conj_bn.ciphertext), bn.delta*2)
                scale_adjust=int(1/(bn.delta*4)/2)
            else:
                scale_adjust=1
            an.ciphertext, bn.ciphertext = engine.auto_level(an.ciphertext, bn.ciphertext)
            bn = Ciphertext(engine.mult_int_scalar(bn.ciphertext, scale_adjust), bn.delta*scale_adjust)
            an = Ciphertext(engine.mult_int_scalar(an.ciphertext, 1), an.delta)
            temp = engine.add(an.ciphertext, engine.imult(bn.ciphertext))
            
            temp1 = engine.bootstrap(engine.mult_int_scalar(temp,2**4))
            temp1 = engine.mult_scalar(temp1,1/2**4/1.2)
            
            conj = engine.conjugate(temp1)
            an.ciphertext = engine.add(temp1, conj)
            bn.ciphertext = engine.imult(engine.sub(conj, temp1))
            
        d=d+1
        kn=2/(en+1)
        an_temp = Ciphertext(engine.negate(engine.add_scalar(bn.ciphertext, -2/kn*bn.delta)), bn.delta)
        an = Ciphertext(engine.auto_ct_ct_mult(an.ciphertext,an_temp.ciphertext), an.delta*an_temp.delta)
        an = Ciphertext(an.ciphertext, an.delta/kn**2)
        
        bn_temp = Ciphertext(engine.negate(engine.add_scalar(bn.ciphertext,-2/kn*bn.delta)), bn.delta)
        bn = Ciphertext(engine.auto_ct_ct_mult(bn.ciphertext,bn_temp.ciphertext), bn.delta*bn_temp.delta)
        bn = Ciphertext(bn.ciphertext, bn.delta/kn**2)
        en=kn*en*(2-kn*en)
                          
        scale_adjust=int(1/bn.delta/2**8)
        if scale_adjust>1:
            conj_an=Ciphertext(engine.conjugate(an.ciphertext), an.delta)
            an=Ciphertext(engine.add(an.ciphertext,conj_an.ciphertext), an.delta*2)
            conj_bn=Ciphertext(engine.conjugate(bn.ciphertext),bn.delta)
            bn=Ciphertext(engine.add(bn.ciphertext,conj_bn.ciphertext), bn.delta*2)
            scale_adjust=int(1/bn.delta/2**8)
        else:
            scale_adjust=1
        
        bn = Ciphertext(engine.mult_int_scalar(bn.ciphertext, scale_adjust), bn.delta*scale_adjust)
        an = Ciphertext(engine.mult_int_scalar(an.ciphertext, scale_adjust), an.delta*scale_adjust)
        
    output_precision = en
    return an.ciphertext, an.delta, output_precision

def he_exp1(engine:CkksEngine, enc_x, min_x, max_x, n):
    # Map the input domain [min_x, max_x] to the symmetric interval [-M/32, M/32]
    mid_x = (min_x+max_x)/2
    enc_x = engine.add_scalar(enc_x, -mid_x/32)

    # Note: Although enc_x is expected to be scaled by 1/4 during the previous matrix multiplication,
    # we intentionally apply a 1/32 scaling instead to increase the magnitude of the polynomial coefficients
    # for better numerical stability and precision during encrypted evaluation.
    
    # - exponential polynomial approximation

    # 1. We approximate exp(4x) using a polynomial over the domain x ∈ [-M/32, M/32].
    #    Due to the 1/32 input scaling, this is effectively equivalent to approximating exp(x)
    #    over the original domain x ∈ [-M/8, M/8].
    #    This scaling strategy amplifies the polynomial coefficients,
    #    which helps reduce the relative impact of ciphertext noise during evaluation.

    # 2. In the softmax computation, we normalize the sum of exp(x_i) values so that the result lies in [0, 1].
    #    To account for this normalization, we pre-scale the polynomial approximation of exp(x)
    #    by embedding the normalization factor directly into the coefficients.

    p = [ 0.032855468333339584, 0.05948672763856172, 0.03881607331549499, 0.0670090353368128, 
            0.15202099984697098, 0.20618261949210986, 0.23721029007596767, 0.26787311936472025, 
            0.27220647178765545, 0.2379982262906916, 0.1780344447042791, 0.11128698173597897, 
            0.05566510463488879, 0.020873931555133732, 0.005218196900295354, 0.0006522770224130905]   
    p.reverse()
    p = np.array(p)
    exp_x = evaluate_polynomial_stockmeyer(engine, p, enc_x)
    for _ in range(int(np.log2(n))):
        exp_x = engine.square(exp_x)        
    return exp_x

def he_exp2(engine:CkksEngine, enc_x, min_x, max_x, n):
    # Map the input domain [min_x, max_x] to the symmetric interval [-M/64, M/64]
    mid_x = (min_x+max_x)/2
    enc_x = engine.add_scalar(enc_x, -mid_x/64)

    # Note: Although enc_x is expected to be scaled by 1/8 during the previous matrix multiplication,
    # we intentionally apply a 1/64 scaling instead to increase the magnitude of the polynomial coefficients
    # for better numerical stability and precision during encrypted evaluation.

    p = [ 0.008201736399899691, 0.014226972463907047, -0.008386712802267769, -0.009262268572236316, 
                    0.0397324053296174, 0.04817928878801878, 0.016604320800445653, 0.02336452059478656, 
                    0.04217318306400685, 0.03517495704921328, 0.022268203231858744, 0.014216636671807894, 
                    0.00749909544008294, 0.0027930565779849003, 0.0006877176070101981, 8.615994668877663e-05] 
    p.reverse()
    p = np.array(p) 
    
    exp_x = evaluate_polynomial_stockmeyer(engine, p, enc_x)
    for _ in range(int(np.log2(n))):
        exp_x = engine.square(exp_x)        
    return exp_x

class Ciphertext:
    def __init__(self, ciphertext:DataStruct, delta:float):
        self.ciphertext = ciphertext
        self.delta = delta
