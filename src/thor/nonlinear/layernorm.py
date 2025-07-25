import numpy as np

from ..ckks import CkksEngine

def he_layernorm1(engine:CkksEngine, x, gamma ,beta ,var_e = 10**(-5),min_var = 0.15,max_var = 10,  debug=False, sk=None):
    return he_layernorm(engine, x, gamma, beta, var_e, min_var, max_var, debug=debug, sk=sk)

def he_layernorm2(engine:CkksEngine, x, gamma ,beta ,var_e = 10**(-5),min_var = 0.2,max_var = 150,  debug=False, sk=None):
    return he_layernorm(engine, x, gamma, beta, var_e, min_var, max_var,n=768, debug=debug, sk=sk)

def he_layernorm3(engine:CkksEngine, x, gamma ,beta ,var_e = 10**(-5),min_var = 0.75,max_var = 2500,  debug=False, sk=None):
    return he_layernorm(engine, x, gamma, beta, var_e, min_var, max_var,n=768, debug=debug, sk=sk)

def he_layernorm(engine:CkksEngine, l,gamma,beta, var_e,min_var, max_var,n=768, debug=False, sk=None):
    if min_var <= 0.16:
        name = 'ln1'
    elif min_var >= 0.74:
        name = 'ln3'
    else:
        name = 'ln2'
    epsilon_var1 = min_var/max_var
    w_buffer=1.05
    max_for_denominator = (max_var*w_buffer+var_e)*n**2
    
    mask = np.array(([1/max_for_denominator**(1/2)]*6+[0]*10)*2**11)
    if name != 'ln1':
        mask = mask/2
    enc_l = [engine.cm_mult(ct, mask) for ct in l]
    
    
    #Compute n * sigma(x)
    sum_x = enc_l[0]
    for i in range(1,len(enc_l)):
        sum_x = engine.add(sum_x, enc_l[i])
    sum_x= engine.rotsum(sum_x, 2**11)
    sum_x=engine.add(sum_x,engine.rotate_left(sum_x,1))
    sum_x=engine.add(sum_x,engine.rotate_left(sum_x,2))
    sum_x=engine.add(sum_x,engine.rotate_left(sum_x,4))
    mask = np.array(([1]*1+[0]*15)*2**11)
    sum_x = engine.cm_mult(sum_x,mask)
    #Compute sigma(x)^2 on the first slot
    sq_sum_x = engine.square(sum_x)  
    #Rotations to Spread the Square of the Sum Across Slots
    sum_x = engine.add(sum_x,engine.rotate_left(sum_x,-1))
    sum_x = engine.add(sum_x,engine.rotate_left(sum_x,-2))
    sum_x = engine.add(sum_x,engine.rotate_left(sum_x,-4))
    
    #Compute numerator n*x - sigma(x)
    nx = [engine.mult_int_scalar(ct, n) for ct in enc_l]
    if nx[0].level_calc > sum_x.level_calc:
        sum_x =engine.level_up(sum_x, nx[0].level_calc)
    numerator = [engine.sub(ct, sum_x) for ct in nx]
    
    #Compute sigma(x**2)
    sigma_x2 = engine.square(enc_l[0])
    for i in range(1,len(enc_l)):
        sigma_x2 = engine.add(sigma_x2, engine.square(enc_l[i]))
    sigma_x2 = engine.rotsum(sigma_x2, 2**11)
    sigma_x2=engine.add(sigma_x2,engine.rotate_left(sigma_x2,1))
    sigma_x2=engine.add(sigma_x2,engine.rotate_left(sigma_x2,2))
    sigma_x2=engine.add(sigma_x2,engine.rotate_left(sigma_x2,4))
    sigma_x2 = engine.cm_mult(sigma_x2,mask)
    
    #Compute variance 
    n_sigma_x2 = engine.mult_int_scalar(sigma_x2, n)
    variance = engine.sub(n_sigma_x2,sq_sum_x)
    variance = engine.add_scalar(variance, var_e/max_for_denominator)
    
    #Encrypt one
    enc_one = engine.encode_and_encrypt(mask, level=variance.level_calc)
    
    #Compute Inverse Square Root
    #Rotations to spread the Inverse Square Root Across Slots
    
    denominator = he_invsqrt(engine, enc_one,variance,epsilon_var1,alpha=0.001,mask=mask)
    if name == 'ln1':
        if denominator.level_calc > 29- 7:
            denominator = engine.bootstrap(denominator)
    else:
        if denominator.level_calc > 29-10:
            denominator = engine.bootstrap(denominator)
    denominator = engine.add(denominator,engine.rotate_left(denominator,-1))
    denominator = engine.add(denominator,engine.rotate_left(denominator,-2))
    denominator = engine.add(denominator,engine.rotate_left(denominator,-4))
    
    #Compute LayerNorm by dividing the numerator by the denominator
    layernorm_x = np.full((8,), None, dtype=object)
    for i in range(4):
        layernorm_x[i] = engine.auto_ct_ct_mult(numerator[i],
                                                engine.rescale(engine.pt_ct_mult(gamma[i],denominator))
                                                )
        layernorm_x[i+4] = engine.auto_ct_ct_mult(numerator[i+4], 
                                                   engine.rescale(engine.pt_ct_mult(gamma[i+4],denominator))
                                                   )

        layernorm_x[i] = engine.pc_add(beta[i], layernorm_x[i])
        layernorm_x[i+4] = engine.pc_add(beta[i+4],layernorm_x[i+4])
        
        # Since encoding is done as 1/2
        layernorm_x[i] = engine.cc_add(layernorm_x[i], layernorm_x[i])  
        layernorm_x[i+4] = engine.cc_add(layernorm_x[i+4], layernorm_x[i+4]) 
    return layernorm_x

def he_invsqrt(engine:CkksEngine, numerator, denominator,e,alpha, mask=np.array(([1]*1+[0]*15)*2**11)):
    d=0
    an=denominator
    bn=numerator
    en=e
    while en<1-alpha:
        d=d+1
        kn=np.roots([1-en**3,6*en**2-6,9-9*en])[1] # find kn s.t. f(kn*en)=f(kn*1)
        bn1=engine.cm_mult(bn, (kn**(3/2)/2)*mask)
        if an.level_calc>(29-4) or bn.level_calc>(29-4):
            an = engine.mult_int_scalar(an,2**6)
            an = engine.bootstrap(an)
            an = engine.mult_scalar(an, 1/2**6)
            bn1 = engine.bootstrap(bn1)
            bn1 = engine.mult_scalar(bn1,2**0)
            
        an, bn1 = engine.auto_level(an, bn1)
        bn2=engine.mc_sub((3/kn)*mask, an)
        bn1, bn2 = engine.auto_level(bn1, bn2)
        bn=engine.auto_ct_ct_mult(bn1,bn2)

        an1=engine.cm_mult(an,(kn**3/4)*mask)
        an2=engine.square(engine.mc_sub((3/kn)*mask,an))
        an=engine.auto_ct_ct_mult(an1,an2) 
        
        en=kn*en*(3-kn*en)**2/4
    return bn