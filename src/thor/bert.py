import numpy as np
from liberate.fhe.data_struct import DataStruct
import torch
from functools import partial
import time

from .linear import ThorLinearEvaluator
from .nonlinear.gelu import he_gelu#, he_gelu2
from .nonlinear.softmax import he_softmax1, he_softmax2
from .nonlinear.layernorm import he_layernorm1, he_layernorm2, he_layernorm3
from .nonlinear.tanh import he_tanh
        
class ThorModule:
    def __init__(self) -> None:
        self.weights = {}
        self.devices = []
        
    def to(self, devices=[0]):
        print(f"Memory allocated to GPU: {torch.cuda.memory_allocated(devices[0]) /1024**3} before layer allocation")

        for key in self.keys:
            for index in np.ndindex(self.weights[key].shape):
                if isinstance(self.weights[key][index], torch.Tensor):
                    encoded = [self.weights[key][index].to(f"cuda:{devices[0]}")]
                    self.weights[key][index] = encoded
        self.devices = devices
                    
    def cpu(self):
        for key in self.keys:
            for index in np.ndindex(self.weights[key].shape):
                if isinstance(self.weights[key][index], list):
                    self.weights[key][index] = self.weights[key][index][0].cpu()          
        self.devices = []

class ThorBert:
    def __init__(self, evaluator:ThorLinearEvaluator, weights: dict[np.ndarray], max_layer_batch=2):
        self.n_layers = 12 
        self.evaluator = evaluator
        self.engine = self.evaluator.engine
        self.attentions:list[ThorBertAttention] = []
        self.ffs:list[ThorBertFF] = []
        for layer in range(self.n_layers):
            self.attentions.append(ThorBertAttention(evaluator, weights, layer))
            self.ffs.append(ThorBertFF(evaluator, weights, layer))
        self.pooler = ThorBertPooler(evaluator, weights)
        self.classifier = ThorBertClassifier(evaluator, weights)
        self.max_layer_batch = max_layer_batch
        self.devices = []
             
    def to(self, devices=[]):
        print(f"Memory allocated to GPU: {torch.cuda.memory_allocated(devices[0]) /1024**3:.2f} GB after pooler and classifier")
        self.devices = devices
    
    def forward(self, x: np.ndarray, attention_mask, devices=[], debug=False, sk = None):
        if debug:
            if sk is None:
                raise ValueError("Please provide secret key")
        if devices == []:
            if self.devices == []:
                raise ValueError("Please set GPU devices")
            devices = self.devices
    
        start_time = time.time()
        if x.shape != (4,):
            raise ValueError("Input of bert should be (8,)")
        
        for i in range(self.n_layers//self.max_layer_batch):
            for j in range(i*self.max_layer_batch, (i+1)*self.max_layer_batch):
                self.attentions[j].to(devices)
                self.ffs[j].to(devices)
                print(f"Memory allocated to GPU: {torch.cuda.memory_allocated(devices[0]) /1024**3:.2f} GB after layer {j} allocation: ")
                
            for j in range(i*self.max_layer_batch, (i+1)*self.max_layer_batch):
                print(f"Forwarding layer: {j}")
                temp = time.time()
                x = self.attentions[j].forward(x, attention_mask, debug=debug, sk=sk)
                x = self.ffs[j].forward(x, debug=debug, sk=sk)
                print(f"Time taken for layer {j} forward: {time.time() - temp:.2f} seconds")
            
            for j in range(i*self.max_layer_batch, (i+1)*self.max_layer_batch):
                self.attentions[j].cpu()
                self.ffs[j].cpu()
                print(f"Memory allocated to GPU: {torch.cuda.memory_allocated(devices[0]) /1024**3:.2f} GB after layer {j} release: ")
                
        self.pooler.to(devices)
        self.classifier.to(devices)
        x = self.pooler.forward(x)
        x = self.classifier.forward(x)
        print(f"Total time taken: {time.time() - start_time}")
        return x
            
class ThorBertAttention(ThorModule):
    def __init__(self, evaluator:ThorLinearEvaluator, weights: dict[np.ndarray[DataStruct]], layer_idx: int):
        self.evaluator = evaluator
        self.engine = self.evaluator.engine
        self.layer_idx = layer_idx
        self.weights = {}
        for qkv in ['query', 'key', 'value']:
            self.weights[f"{qkv}.weight"] = weights[f'bert.encoder.layer.{layer_idx}.attention.self.{qkv}.weight']
            self.weights[f"{qkv}.bias"] = weights[f'bert.encoder.layer.{layer_idx}.attention.self.{qkv}.bias']
        self.weights['dense.weight'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.dense.weight']
        self.weights['dense.bias'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.dense.bias']
        self.weights['LayerNorm.weight'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.LayerNorm.weight']
        self.weights['LayerNorm.bias'] = weights[f'bert.encoder.layer.{layer_idx}.attention.output.LayerNorm.bias']
        
        self.keys = ['query.weight', 'query.bias', 
                         'key.weight', 'key.bias', 
                         'value.weight', 'value.bias', 
                         'dense.weight', 'dense.bias',
                         'LayerNorm.weight', 'LayerNorm.bias']

        self.softmax = partial(he_softmax2, engine=self.engine) if layer_idx == 2 else partial(he_softmax1, engine=self.engine)
        self.layernorm = partial(he_layernorm1, engine=self.engine, gamma=self.weights['LayerNorm.weight'], beta=self.weights['LayerNorm.bias'])
        self.devices = []
        
    def forward(self, x: np.ndarray, attention_mask:np.ndarray, debug=False, sk=None):
        if x.shape == (8,):
            x_cplx = np.full((4,), None, dtype=object)
            for i in range(4):
                x_cplx[i] = self.engine.cc_add(x[i], self.engine.imult(x[i+4]))
                x_cplx[i] = self.engine.cc_add(x_cplx[i], self.engine.rotate_left(x_cplx[i], -6))
        
        elif x.shape == (4,):
            x_cplx = x
            x = np.full((8,), None, dtype=object)
            for i in range(4):
                conj = self.engine.conjugate(x_cplx[i])
                x[i] = self.engine.mult_scalar(self.engine.cc_add(x_cplx[i], conj), 1/2)
                x[i+4] = self.engine.mult_scalar(self.engine.imult(self.engine.cc_sub(conj, x_cplx[i])), 1/2)
                x_cplx[i] = self.engine.level_up(x_cplx[i], 21)

        
        input_lev = x_cplx[0].level_calc
        
        x_cplx_rots = self.evaluator.make_rotated_copies(x_cplx)
            
        q = self.query(x_cplx_rots)
        k = self.key(x_cplx_rots)
        v = self.value(x_cplx_rots)
        
        l_k = self.evaluator.transpose_upper_to_lower(k)
        #Complexify l_k
        l_k_cplx = np.full((4,), None, dtype=object)
        for i in range(4):
            l_k_cplx[i] = self.engine.cc_add(self.engine.level_up(l_k[i], l_k[i].level_calc+1), self.engine.imult(self.evaluator.rotate_internal(l_k[i], 64, mode='att')))
            # Pre rescale
            l_k_cplx[i] = self.engine.rescale(l_k_cplx[i]) #Scale: delta

        for i in range(4):
            q[i] = self.engine.level_up(q[i], q[i].level_calc+1)
            #Pre rescale
            q[i] = self.engine.rescale(q[i]) #Scale: delta
        q_copies = self.evaluator.make_copies(q)#, merge_copy=True)
        sftmx_scale = 1
        att_score = self.calculate_attention_score(l_k_cplx, q_copies, bootstrap=False, scale=sftmx_scale, rescale=False)
        for i in range(4):
            temp = self.engine.cc_add(att_score[i], self.engine.imult(att_score[i+4]))
            temp = self.engine.bootstrap(temp)
            conj = self.engine.conjugate(temp)
            att_score[i] = self.engine.cc_add(temp, conj)
            att_score[i+4] = self.engine.imult(self.engine.cc_sub(conj, temp))
        
        att_prob = self.softmax(x=att_score, attention_mask=attention_mask, rescale=False, debug=debug, sk=sk) #Returns att_prob, already copied.
        v_cplx = np.full((2,), None, dtype=object)
        for i in range(2):
            v_cplx[i] = self.engine.cc_add(v[i], self.engine.imult(v[i+2]))

        if att_prob[0].level_calc < v_cplx[0].level_calc:
            for j in range(128):
                att_prob[j] = self.engine.level_up(att_prob[j], v[0].level_calc)
        elif att_prob[0].level_calc > v_cplx[0].level_calc:
            for j in range(2):
                v_cplx[j] = self.engine.level_up(v_cplx[j], att_prob[0].level_calc)

        for i in range(2):
            # Pre rescale
            v_cplx[i] = self.engine.rescale(v_cplx[i]) #Scale: delta
        for j in range(128):
            att_prob[j] = self.engine.rescale(att_prob[j])        
            
        att_context = self.calculate_attention_context(v_cplx, att_prob) #Scale: delta^2
        
        for i in range(2):
            att_context[i] = self.engine.bootstrap(att_context[i])
        att_context_rots = self.evaluator.make_rotated_copies(att_context)
        
        dense_output = self.dense(att_context_rots)
        
        x_out_sum = np.full((8,), None, dtype=object)

        for i in range(4):
            x_out_sum[i] = self.engine.add(x[i], dense_output[i])
            x_out_sum[i+4] = self.engine.add(x[i+4], dense_output[i+4])
 
        x = self.layernorm(x=x_out_sum, debug=debug, sk=sk) #-> bs 1, level: 19
        return x
    
    def query(self, x: np.ndarray[DataStruct]):
        """"
        Calculates the query layer of the self attention
        @param x: Complexified lower diagonals of the input, ciphertext array of shape (64,)
        @return: Lower diagonals of the query layer, ciphertext array of shape (4,)
        """
        if x.shape != (64,):
            raise ValueError("Input of self attention should be (64,)")
        wx = self.evaluator.pt_ct_matmul(self.weights['query.weight'], x)
        if wx.shape[0] != 4:
            raise ValueError("Shape of wx should be (4,)")
        
        q = np.full((4,), None, dtype=DataStruct)
        for i in range(4):
            q[i] = self.engine.pc_add(self.weights['query.bias'][i], wx[i])
            q[i] = self.engine.cc_add(q[i], self.engine.conjugate(q[i]))
        return q
        
    def key(self, x: np.ndarray[DataStruct]):
        """
        Calculates the key layer of the self attention
        @param x: Complexified lower diagonals of the input, ciphertext array of shape (64,)
        @return: Lower diagonals of the key layer, ciphertext array of shape (4,)
        """
        if x.shape != (64,):
            raise ValueError("Input of self attention should be (64,)")
        wx = self.evaluator.pt_ct_matmul(self.weights['key.weight'], x)
        if wx.shape[0] != 4:
            raise ValueError("Shape of wx should be (4,)")
        for i in range(4):
            wx[i] = self.engine.pc_add(self.weights['key.bias'][i], wx[i])
            wx[i] = self.engine.cc_add(wx[i], self.engine.conjugate(wx[i]))
        return wx
    
    def value(self, x: np.ndarray[DataStruct]):
        """
        Calculates the value layer of the self attention
        @param x: Complexified lower diagonals of the input, ciphertext array of shape (64,)
        @return: Lower diagonals of the value layer, ciphertext array of shape (4,)
        """
        if x.shape != (64,):
            raise ValueError("Input of self attention should be (64,)")
        wx = self.evaluator.pt_ct_matmul(self.weights['value.weight'], x)
        if wx.shape[0] != 4:
            raise ValueError("Shape of wx should be (4,)")
        for i in range(4):
            wx[i] = self.engine.pc_add(self.weights['value.bias'][i], wx[i])
            wx[i] = self.engine.cc_add(wx[i], self.engine.conjugate(wx[i]))
        return wx
    
    def calculate_attention_score(self, k_cplx: np.ndarray[DataStruct], q_copies: np.ndarray[DataStruct], scale=1, bootstrap=True, rescale=False, debug=False):
        """
        Calculates the attention score(qk^T)
        @param k_cplx: key, ciphertext array of shape (4,) [ [l0 +il64| l1 | ...l15+il79], ...[l48+il112| l49+il113|...l63+il127] ]
        @param q_copies: query, ciphertext array of shape (64,) [ [l0 | l0| ...], [l1...] ... [l64...] ]
        @return: output, ciphertext array of shape (8,)
        """
        if k_cplx.shape != (4,) or q_copies.shape != (64,):
            raise ValueError("Shape of k_cplx should be (4,) and q_copies should be (64,)")
        
        if k_cplx[0].level_calc != q_copies[0].level_calc:
            if k_cplx[0].level_calc > q_copies[0].level_calc:
                for i in range(len(q_copies)):
                    q_copies[i]=self.engine.level_up(q_copies[i],k_cplx[0].level_calc)
            else:
                for i in range(len(k_cplx)):
                    k_cplx[i]=self.engine.level_up(k_cplx[i],q_copies[0].level_calc)
            # raise ValueError("Levels of k_cplx and q_copies should be the same")
            
        n_in = 64
        n_out = 4
        lev = k_cplx[0].level_calc #Level: l
        
        ttemp = np.full((n_out,4), None, dtype=DataStruct)
        
        if bootstrap:
            scale = scale / 2
        scales = np.full((2**15,),scale, dtype=float)
        scale_pt = self.engine.encode(scales, level=lev)
        
        masks = self.evaluator.masks['ct_ct_matmul']
        
        #n = 0
        for out in range(n_out):
            temp_ct_ct = self.engine.ct_ct_mult(k_cplx[out], q_copies[0], rescale=rescale, relin=False)
            ttemp[out, 0] = self.engine.pt_ct_mult_extended(scale_pt, temp_ct_ct)                      
            
        for n in range(1, n_in):
            q = (n // 16)
            j = n % 16
            rot = n
            l_prime = q_copies[n]
            temp = np.full((n_out,4), None, dtype=DataStruct)
            rrot = (2**11)*j - 16 * rot
            rrot = rrot%(2**15)

            if j == 0:
                for out in range(n_out):
                    rot_temp = self.engine.rotate_left(k_cplx[out], -rrot, self.engine.rot_keys[(self.engine.num_slots-rrot)])
                    temp_ct_ct = self.engine.ct_ct_mult(rot_temp, l_prime, rescale=rescale, relin=False)

                    for i in range(1):
                        temp[out, i] = self.engine.pt_ct_mult_extended(masks[i][n], temp_ct_ct)
                    scaled_temp_ct_ct = self.engine.mult_int_scalar_triplet(temp_ct_ct, 2**41)
                    temp[out, 1] = self.engine.sub(scaled_temp_ct_ct, temp[out,0])
                
                if q == 1:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[3,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[3,1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0,0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0,1])
                    
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[1,0])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[1,1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[2,0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[2,1])
                    
                elif q == 2:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[2,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[2,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[3,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[3,1])
                    
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[0,0])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[0,1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[1,0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[1,1])
                    
                elif q == 3:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[2,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[2,1])
                    
                    ttemp[2,2] = self.engine.cc_add(ttemp[2,2], temp[3,0])
                    ttemp[2,3] = self.engine.cc_add(ttemp[2,3], temp[3,1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[0,0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[0,1])

            else:  
                for out in range(n_out):
                    rot_temp = self.engine.rotate_left(k_cplx[out], -rrot, self.engine.rot_keys[self.engine.num_slots-rrot])
                    temp_ct_ct = self.engine.ct_ct_mult(rot_temp, l_prime, rescale=rescale, relin=False)
                    for i in range(3):
                        temp[out, i] = self.engine.pt_ct_mult_extended(masks[i][n], temp_ct_ct)
                    add_temp = self.engine.add(self.engine.add(temp[out,0], temp[out,1]), temp[out,2])
                    scaled_temp_ct_ct = self.engine.mult_int_scalar_triplet(temp_ct_ct, 2**41)
                    temp[out, 3] = self.engine.sub(scaled_temp_ct_ct, add_temp)
                
                if q == 0:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[3, 2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[3, 3])
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[0, 0])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[0, 1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0, 2])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0, 3])
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[1, 0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[1, 1])
                    
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[1, 2])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[1, 3])
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[2, 0])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[2, 1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[2, 2])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[2, 3])
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[3, 0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[3, 1])
                    
                
                elif q == 1:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[2, 2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[2, 3])
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[3, 0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[3, 1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[3, 2])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[3, 3])
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0, 0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0, 1])
                    
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[0, 2])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[0, 3])
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[1, 0])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[1, 1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[1, 2])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[1, 3])
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[2, 0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[2, 1])
                    
                elif q == 2:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1, 2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1, 3])
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[2, 0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[2, 1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[2, 2])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[2, 3])
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[3, 0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[3, 1])
                    
                    ttemp[2,2] = self.engine.cc_add(ttemp[2,2], temp[3, 2])
                    ttemp[2,3] = self.engine.cc_add(ttemp[2,3], temp[3, 3])
                    ttemp[2,0] = self.engine.cc_add(ttemp[2,0], temp[0, 0])
                    ttemp[2,1] = self.engine.cc_add(ttemp[2,1], temp[0, 1])
                    
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[0, 2])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[0, 3])
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[1, 0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[1, 1])
                    
                elif q == 3:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[0, 2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[0, 3])
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1, 0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1, 1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[1, 2])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[1, 3])
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[2, 0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[2, 1])
                    
                    ttemp[2,2] = self.engine.cc_add(ttemp[2,2], temp[2, 2])
                    ttemp[2,3] = self.engine.cc_add(ttemp[2,3], temp[2, 3])
                    ttemp[2,2] = self.engine.cc_add(ttemp[2,2], temp[3, 0])
                    ttemp[2,3] = self.engine.cc_add(ttemp[2,3], temp[3, 1])
                    
                    ttemp[3,2] = self.engine.cc_add(ttemp[3,2], temp[3, 2])
                    ttemp[3,3] = self.engine.cc_add(ttemp[3,3], temp[3, 3])
                    ttemp[3,0] = self.engine.cc_add(ttemp[3,0], temp[0, 0])
                    ttemp[3,1] = self.engine.cc_add(ttemp[3,1], temp[0, 1])
        
        output = np.full((8,), None, dtype=DataStruct)
        ttemp_relin = np.full((n_out,4), None, dtype=DataStruct)
        output_cplx = np.full((4,), None, dtype=DataStruct)
        output_cplx_scaled = np.full((4,), None, dtype=DataStruct)

        for out in range(n_out):
            for i in range(4):
                ttemp_relin[out, i] = self.engine.relinearize(ttemp[out, i])
            ttemp_relin[out, 1] = self.engine.rotate_left(ttemp_relin[out, 1], -2**11)
            ttemp_relin[out, 3] = self.engine.rotate_left(ttemp_relin[out, 3], -2**11)
            ttemp_relin[out, 2] = self.engine.imult(self.engine.conjugate(self.engine.cc_add(ttemp_relin[out, 2], ttemp_relin[out, 3])))
            output_cplx[out] = (self.engine.cc_add(self.engine.cc_add(ttemp_relin[out, 0], ttemp_relin[out, 1]), ttemp_relin[out, 2]))
            output_cplx_scaled[out] = self.engine.rescale(output_cplx[out]) #Scale: delta^2
            if bootstrap:
                output_cplx_scaled[out] = self.engine.bootstrap(output_cplx_scaled[out])
            conj = self.engine.conjugate(output_cplx_scaled[out])
            output[out] = (self.engine.cc_add(output_cplx_scaled[out], conj))
            output[out+n_out] = (self.engine.imult(self.engine.cc_sub(conj, output_cplx_scaled[out])))     
        return output
    
    def calculate_attention_context(self, l_vt_cplx: np.ndarray[DataStruct], l_at_copies: np.ndarray[DataStruct], scale=1, complex=True, rescale=False):
        """
        @param l_vt_cplx: value, ciphertext array of shape (2,)
        @param l_at_copies: attention prob, ciphertext array of shape (128,)
        """
        lev = l_at_copies[0].level_calc
        if l_vt_cplx[0].level_calc != lev:
            raise ValueError("Level of l_vt_cplx should be the same as l_at_copies")
        
        n_in = 128
        n_out = 2

        ttemp = np.full((n_out,4), None, dtype=DataStruct)
        
        if not complex:
            scale = scale/2
            
        scales = np.full((2**15,),scale, dtype=float)
        scale_pt = self.engine.encode(scales, level=lev)
        masks = self.evaluator.masks['ct_ct_matmul']

    
        for out in range(n_out):            
            temp_ct_ct = self.engine.ct_ct_mult(l_vt_cplx[out], l_at_copies[0], rescale=rescale, relin=False)
            ttemp[out, 0] = self.engine.pt_ct_mult_extended(scale_pt, temp_ct_ct)
            
        for n in range(1, n_in):
            q = (n // 16) % 4  
            j = n % 16
            rot = n
            l_prime = l_at_copies[n]
            
            temp = np.full((n_out,4), None, dtype=DataStruct)

            rrot = (2**11)*j - 16 * rot
            rrot = rrot%(2**15)
            
            if j == 0:
                for out in range(n_out):
                    rot_temp = self.engine.rotate_left(l_vt_cplx[out], -rrot)
                    temp_ct_ct = self.engine.ct_ct_mult(rot_temp, l_prime, rescale=rescale, relin=False)

                    for i in range(1):
                        temp[out, i] = self.engine.pt_ct_mult_extended(masks[i][n], temp_ct_ct)
                    scaled_temp_ct_ct = self.engine.mult_int_scalar_triplet(temp_ct_ct, 2**41)
                    temp[out, 1] = self.engine.sub(scaled_temp_ct_ct, temp[out,0])
    
                if q == 1:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1,1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0,0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0,1])
                                    
                elif q == 2:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[0,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[0,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[1,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[1,1])
                    
                elif q == 3:
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[1,0])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[1,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[0,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[0,1])
                    
                elif q == 0:  
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[0,0])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[0,1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[1,0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[1,1])


            elif j != 0:            
                
                for out in range(n_out):
                    rot_temp = self.engine.rotate_left(l_vt_cplx[out], -rrot)
                    temp_ct_ct = self.engine.ct_ct_mult(rot_temp, l_prime, rescale=rescale, relin=False)                   
                    for i in range(3):
                        temp[out, i] = self.engine.pt_ct_mult_extended(masks[i][n], temp_ct_ct)
                    add_temp = self.engine.add(self.engine.add(temp[out,0], temp[out,1]), temp[out,2])
                    scaled_temp_ct_ct = self.engine.mult_int_scalar_triplet(temp_ct_ct, 2**41)
                    temp[out, 3] = self.engine.sub(scaled_temp_ct_ct, add_temp)
                
                if q == 0:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1,2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1,3])
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[0,0])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[0,1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0,2])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0,3])
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[1,0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[1,1])

                elif q == 1:
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[0,2])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[0,3])
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[1,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[1,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[1,2])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[1,3])           
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[0,0])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[0,1])
                    
                elif q == 2:
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[1,2])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[1,3])
                    ttemp[0,2] = self.engine.cc_add(ttemp[0,2], temp[0,0])
                    ttemp[0,3] = self.engine.cc_add(ttemp[0,3], temp[0,1])
                    
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[0,2])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[0,3])
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[1,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[1,1])
     
                elif q == 3:
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[0,2])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[0,3])
                    ttemp[0,0] = self.engine.cc_add(ttemp[0,0], temp[1,0])
                    ttemp[0,1] = self.engine.cc_add(ttemp[0,1], temp[1,1])
                    
                    ttemp[1,0] = self.engine.cc_add(ttemp[1,0], temp[1,2])
                    ttemp[1,1] = self.engine.cc_add(ttemp[1,1], temp[1,3])
                    ttemp[1,2] = self.engine.cc_add(ttemp[1,2], temp[0,0])
                    ttemp[1,3] = self.engine.cc_add(ttemp[1,3], temp[0,1])
       
        output_cplx = np.full((2,), None, dtype=DataStruct)
        output_cplx_scaled = np.full((2,), None, dtype=DataStruct)
        ttemp_relin = np.full((n_out,4), None, dtype=DataStruct)
        
        for out in range(n_out):
            for i in range(4):
                ttemp_relin[out, i] = self.engine.relinearize(ttemp[out, i])
            ttemp_relin[out, 1] = self.engine.rotate_left(ttemp_relin[out, 1], -2**11)
            ttemp_relin[out, 3] = self.engine.rotate_left(ttemp_relin[out, 3], -2**11)
            ttemp_relin[out, 2] = self.engine.imult(self.engine.conjugate(self.engine.cc_add(ttemp_relin[out, 2], ttemp_relin[out, 3])))
            output_cplx[out] = self.engine.cc_add(self.engine.cc_add(ttemp_relin[out, 0], ttemp_relin[out, 1]), ttemp_relin[out, 2])
            output_cplx_scaled[out] = self.engine.rescale(output_cplx[out])
        return output_cplx_scaled
    
    
    def dense(self, x: np.ndarray[DataStruct]):
        """
        @param x: input, ciphertext array of shape (32,)
        @return: output, ciphertext array of shape (8,)
        """

        lev = x[0].level_calc
        wx = self.evaluator.pt_ct_matmul(self.weights['dense.weight'], x) #Level: l -> l + 2
        if wx.shape[0] != 8:
            raise ValueError("Shape of wx should be (8,)")
                
        mask1 = np.ones((self.engine.num_slots,), dtype=int)
        mask1[np.arange(self.engine.num_slots) % (16) < 6] = 0
        
        for i in range(8):
            temp = self.engine.rotate_left(self.engine.mc_mult(
                mask1, wx[i]), 6) #Level: l + 3, scale: delta
            wx[i] = self.engine.cc_add(self.engine.level_up(wx[i], lev+3), temp)
            wx[i] = self.engine.pc_add(self.weights['dense.bias'][i], wx[i])
            wx[i] = self.engine.cc_add(wx[i], self.engine.conjugate(wx[i]))
        return wx
    

class ThorBertFF(ThorModule):
    def __init__(self, evaluator:ThorLinearEvaluator, weights: dict[np.ndarray], layer_idx: int):
        self.evaluator = evaluator
        self.engine = evaluator.engine
        self.weights = {}
        self.weights['dense1.weight'] = weights[f'bert.encoder.layer.{layer_idx}.intermediate.dense.weight']
        self.weights['dense1.bias'] = weights[f'bert.encoder.layer.{layer_idx}.intermediate.dense.bias']
        self.weights['dense2.weight'] = weights[f'bert.encoder.layer.{layer_idx}.output.dense.weight']
        self.weights['dense2.bias'] = weights[f'bert.encoder.layer.{layer_idx}.output.dense.bias']
        self.weights['LayerNorm.weight'] = weights[f'bert.encoder.layer.{layer_idx}.output.LayerNorm.weight']
        self.weights['LayerNorm.bias'] = weights[f'bert.encoder.layer.{layer_idx}.output.LayerNorm.bias']
        self.keys = ['dense1.weight', 'dense1.bias',
                     'dense2.weight', 'dense2.bias',
                     'LayerNorm.weight', 'LayerNorm.bias'
                     ]

        self.gelu = partial(he_gelu, engine=self.engine)
        if layer_idx == 9 or layer_idx == 10:
            self.layernorm = partial(he_layernorm3, engine=self.engine, gamma=self.weights['LayerNorm.weight'], beta=self.weights['LayerNorm.bias'])
        else:
            self.layernorm = partial(he_layernorm2, engine=self.engine, gamma=self.weights['LayerNorm.weight'], beta=self.weights['LayerNorm.bias'])

        self.devices = []
        
    def forward(self, x: np.ndarray, debug=False, sk=None):
        if x.shape != (8,):
            raise ValueError("Input of ff layer should be (8,)")
        
        l = np.full((64,), None,dtype=DataStruct)
        mask = np.full((self.engine.num_slots,), 1, dtype=int)
        mask[np.arange(self.engine.num_slots) % (16) >= 6] = 0
        for i in range(4):
            temp = self.engine.cc_add(x[i], self.engine.imult(x[i+4]))
            temp = self.engine.mc_mult(mask, temp)
            l[16*i] = self.engine.cc_add(temp, self.engine.rotate_left(temp, -8))
            for j in range(1, 16):
                #Change this to hrot later
                index = 16*i+j
                l[index] = self.engine.rotate_left(l[index-1], 2**11)
        
        intermediate_output = self.dense1(l)
        for i in range(8):
            temp = self.engine.cc_add(intermediate_output[0,i], self.engine.imult(intermediate_output[1,i]))
            temp = self.engine.mult_scalar(temp, 1/2)
            temp = self.engine.bootstrap(temp)
            conj = self.engine.conjugate(temp)
            intermediate_output[0,i] = self.engine.cc_add(temp, conj)
            intermediate_output[1,i] = self.engine.imult(self.engine.cc_sub(conj, temp))
        
        gelu_output = self.gelu(x=intermediate_output, sk=sk)
        
        dense2_out = self.dense2(gelu_output) 
        
        ln2_in = np.full((8,), None, dtype=object)
        for i in range(8):
            ln2_in[i] = self.engine.add(x[i], dense2_out[i])
        for i in range(4):
            temp = self.engine.cc_add(ln2_in[i], self.engine.imult(ln2_in[i+4]))
            temp = self.engine.bootstrap(temp)
            conj = self.engine.conjugate(temp)
            ln2_in[i] = self.engine.cc_add(temp, conj)
            ln2_in[i+4] = self.engine.imult(self.engine.cc_sub(conj, temp)) 
            
        ln2_out = self.layernorm(x=ln2_in, debug=debug, sk=sk) 
        if ln2_out[0].level > 8:
            for i in range(8):
                ln2_out[i] = self.engine.level_up(ln2_out[i], 21)
        return ln2_out
        
    def dense1(self, x: np.ndarray): #Matmul + bias
        """
        Input: ciphertext array of shape (64,), complexified
        Returns ciphertext array of shape (2,8)
        """
        #Shape of self.weights['dense1.weight'] is (2, 8, 6, 64)

        if x.shape != (64,):
            raise ValueError("Input of ff dense1 should be (64,)")
        
        wx = np.full((2,8), None, dtype=DataStruct)
        wx[0] = self.evaluator.pt_ct_matmul(self.weights['dense1.weight'][0], x, mode='block_diag_2') #Level: l -> l + 2
        wx[1] = self.evaluator.pt_ct_matmul(self.weights['dense1.weight'][1], x, mode='block_diag_2')
        for i in range(2):
            for j in range(8):
                wx[i,j] = self.engine.pc_add(self.weights['dense1.bias'][i, j], wx[i,j])
                wx[i,j] = self.engine.cc_add(wx[i, j], self.engine.conjugate(wx[i, j]))
        return wx
    
    def dense2(self, x: np.ndarray, debug=False): #Matmul + bias
        """
        Input: ciphertext array of shape (2,8)
        @Return: ciphertext array of shape (8,)
        """
        if x.shape != (2,8):
            raise ValueError("Ouptut of ff1 should be (2,8)")
        l = np.full((2, 64), None, dtype=DataStruct)
        wx = np.full((2, 8), None, dtype=DataStruct)
        x_out = np.full((8,), None, dtype=DataStruct)
        result = np.full((8,), None, dtype=DataStruct)
        for n in range(2):
            for i in range(4):
                l[n, 16*i] = self.engine.cc_add(x[n, i], self.engine.imult(x[n, i+4]))
                for j in range(1, 16):
                    index = 16*i + j 
                    #Change this to hrot later
                    l[n, index] = self.engine.rotate_left(l[n, index-1], 2**11)    
            wx[n] = self.evaluator.pt_ct_matmul(self.weights['dense2.weight'][n], l[n], mode='block_diag_2')
            
        mask = np.ones((self.engine.num_slots,), dtype=int)
        mask[np.arange(self.engine.num_slots) % (16) >= 6] = 0
        for i in range(8):
            x_out[i] = self.engine.cc_add(wx[0,i], wx[1,i])
            x_out[i] = self.engine.cc_add(x_out[i], self.engine.rotate_left(x_out[i], 8))
            result[i] = self.engine.pc_add(self.weights['dense2.bias'][i], x_out[i])
            result[i] = self.engine.cc_add(result[i], self.engine.conjugate(result[i]))
        if debug:
            for i in range(8):
                x_out[i] = self.engine.cc_add(x_out[i], self.engine.conjugate(x_out[i]))
            return result, x_out
        return result
    
class ThorBertPooler(ThorModule):
    def __init__(self, evaluator:ThorLinearEvaluator, weights: dict[np.ndarray]):
        self.evaluator = evaluator
        self.engine = evaluator.engine
        self.weights = {}
        self.weights['dense.weight'] = weights['bert.pooler.dense.weight']
        self.weights['dense.bias'] = weights['bert.pooler.dense.bias']
        self.keys = ['dense.weight', 'dense.bias']
        if not set(self.keys).issubset(set(self.weights.keys())):
            raise ValueError(f"Attention weights should include {self.keys}")
        self.devices = []
        self.tanh = partial(he_tanh, engine=self.engine)

    def forward(self, x: np.ndarray):
        if x.shape != (8,):
            raise ValueError("Input of pooler should be (8,), Shape of x is ", x.shape)
        x_cplx = np.full((4,), None, dtype=DataStruct)
        for i in range(4):
            x_cplx[i] = self.engine.cc_add(x[i], self.engine.imult(x[i+4]))
        x = self.dense(x_cplx) 
        x[0] = self.engine.mult_scalar(x[0], 1/40)
        x = self.tanh(x=x)
        return x
        
    def dense(self, x: np.ndarray, scale=1): #Matmul + bias
        """
        Input: ciphertext array of shape (4,), complexified. [l0+il64| l1+il65| ... | l15 + il79], [l16+il80| l17+il81| ... | l31 + il95] ...
        Returns ciphertext array of shape (2,)
        """
        #Shape of self.weights['dense.weight'] is (2, 8, 6, 64)
        
        w = self.weights['dense.weight']
        lev = x[0].level_calc
        mask = np.zeros((self.engine.num_slots,), dtype=int)
        mask[np.arange(self.engine.num_slots) % (2**11) < 6] = scale
        
        for index, ct in enumerate(x):
            ct_masked = self.engine.mc_mult(mask, ct) #Level: l -> l + 1
            for i in range(7):
                ct_rot = self.engine.rotate_left(ct_masked, -16*2**i)
                ct_masked = self.engine.cc_add(ct_masked, ct_rot)
            x[index] = ct_masked
            
        wx_l = np.full((6,), None, dtype=DataStruct)
        for l in range(6):
            temp = self.engine.pt_ct_mult(w[l, 0], x[0])  #Level: l + 1, scale: delta^2
            for i in range(1,4):
                temp = self.engine.cc_add(temp, self.engine.pt_ct_mult(w[l, i], x[i])) #Level: l + 1, scale: delta^2
            wx_l[l] = self.engine.rescale(temp) #Level: l + 2, scale: delta
            
        temp2 = self.engine.level_up(wx_l[0], lev+3)
        for l in range(1, 6):
            rotated = self.evaluator.rotate_internal(wx_l[l], delta=6-l, mode='block_diag_2') #Level: l +2 -> l + 3
            temp2 = self.engine.cc_add(temp2, rotated)
            
        temp2 = self.evaluator.rotsum(temp2, 2**11)

        wx = np.ndarray((1,), dtype=DataStruct)
        wx[0] = self.engine.pc_add(self.weights['dense.bias'][0], temp2)
        wx[0] = self.engine.cc_add(wx[0], self.engine.conjugate(wx[0]))
        return wx
    
class ThorBertClassifier(ThorModule):
    def __init__(self, evaluator:ThorLinearEvaluator, weights: dict[np.ndarray]):
        self.evaluator = evaluator
        self.engine = evaluator.engine
        self.weights = {}
        cls_name =  "cls.seq_relationship" if 'cls.seq_relationship.weight' in weights.keys() else "classifier"
        self.weights['dense.weight'] = weights[f'{cls_name}.weight']
        self.weights['dense.bias'] = weights[f'{cls_name}.bias']
        self.keys = ['dense.weight', 'dense.bias']
        if not set(self.keys).issubset(set(self.weights.keys())):
            raise ValueError(f"Attention weights should include {self.keys}")
        self.devices = []
                    
    def forward(self, x: np.ndarray):
        if x.shape != (1,):
            raise ValueError("Input of classifier should be (1,)")
        x = self.dense(x)
        return x
    
    def dense(self, x: np.ndarray):
        """
        Input: ciphertext array of shape (1,)
        """
        if x.shape != (1,):
            raise ValueError("Ouptut of pooler should be (1,)")
        n_cls = self.weights['dense.weight'].shape[0]
        wx = np.full((n_cls,), None, dtype=DataStruct)
        for i in range(n_cls):
            wx[i] = self.engine.pt_ct_mult(self.weights['dense.weight'][i], x[0])
            temp = self.engine.cc_add(wx[i], self.engine.rotate_left(wx[i], 1))
            temp2 = self.engine.cc_add(temp, self.engine.rotate_left(temp, 2))
            wx[i] = self.engine.cc_add(temp2, self.engine.rotate_left(temp, 4))
            
            for j in range(4, 11):
                wx[i] = self.engine.cc_add(wx[i], self.engine.rotate_left(wx[i], 2**j))
            wx[i] = self.engine.rescale(wx[i]) #Level: l + 1
            wx[i] = self.engine.pc_add(self.weights['dense.bias'][i], wx[i])
        return wx