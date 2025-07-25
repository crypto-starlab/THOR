import numpy as np
from liberate.fhe.data_struct import DataStruct

from .ckks import CkksEngine

class ThorLinearEvaluator():
    def __init__(self, engine:CkksEngine):
        self.engine = engine
        self.masks = {}
        self.pre_encode_masks()
                
    #Linear Operations
    def pt_ct_matmul(self, w_t:np.ndarray[DataStruct], x_t: np.ndarray[DataStruct], mode ='block_diag_1') -> np.ndarray[DataStruct]:
        """
        @param w_t: Array of model weight plaintexts. Array shape: (n_out_packed, n_diag, n_in_complex)
        @param x_t: Array of model input ciphetexts. Array shape: (n_in_complex,)
        @param mode: 'block_diag_1' or 'block_diag_2'
        'block_diag_1' for qkv(linear projection) and attention_output
        'block_diag_2' for feed forward layer
        @return: wx.T as array of ciphertexts. Array shape: (n_out_packed, )
        """
        if w_t.shape[-1] != x_t.shape[0]:
            raise ValueError(f"n_in of W and x should be equal. n_in of w_t, x_t: {w_t.shape} {x_t.shape}")
        lev = x_t[0].level_calc
        n_diag = w_t.shape[-2]
        n_out_packed = w_t.shape[-3]
        # (a)Parallel diagonal matrix multiplication of submatrices
        ct_submatrices = self.parallel_diagonal_pt_ct_mult(w_t, x_t) #Level: l -> l+1 Shape: (n_out_packed, ll)
        ct_out = np.full((n_out_packed,), None, dtype=DataStruct)
        for out in range(ct_submatrices.shape[-2]):
            ct_temp = self.engine.level_up(ct_submatrices[out, 0], lev +2)
            #(b) rotate internally to align the slots
            for l in range(1, n_diag):
                if mode == 'block_diag_1':
                    rotated = self.rotate_internal(ct_submatrices[out, l], delta=12-l, mode=mode)
                elif mode == 'block_diag_2':
                    rotated = self.rotate_internal(ct_submatrices[out, l], delta=6-l, mode=mode)
                #(c) sum 
                ct_temp = self.engine.cc_add(ct_temp, rotated)
            ct_out[out] = ct_temp
        return ct_out

    def parallel_diagonal_pt_ct_mult(self, w_t:np.ndarray[DataStruct], x_t: np.ndarray[DataStruct]):
        """
        Calculates the block (lower) diagonals ciphertexts of WX.T
        @param w_t: Array of model weights plaintexts. Array shape: (n_out_packed, n_diag, n_in_complex)
        @param x_t: Array of model input ciphetexts. Array shape: (n_in_complex,)
        @return: Array of lower diagonal ciphertexts. Array shape: (n_out_packed, n_diag)
        """
        n_in_c = w_t.shape[-1]
        n_out_packed = w_t.shape[-3]
        ll = w_t.shape[-2]
        ct_diags = np.full((n_out_packed, ll), None, dtype=DataStruct)
        for out in range(n_out_packed):
            for l in range(ll):
                ct_temp = self.engine.pt_ct_mult(w_t[out, l, 0], x_t[(16*out)%n_in_c]) #Level: l, scale: delta^2
                for n in range(1, n_in_c):
                    ct = x_t[(16*out + n) % n_in_c]
                    pt = w_t[out, l, n] 
                    ct_temp = self.engine.cc_add((self.engine.pt_ct_mult(pt, ct)), ct_temp) #Level: l, scale: delta^2
                ct_diags[out, l] = self.engine.rescale(ct_temp) #Level: l + 1, scale: delta
        return ct_diags
    
    def transpose_upper_to_lower(self, u:np.ndarray[DataStruct]) -> np.ndarray[DataStruct]:
        """
        Transpose the encrypted matrix(upper diagonal cts -> lower diagonal cts)
        @param u: Array of upper diagonal ciphertexts
        @return: Array of lower diagonal ciphertexts
        """        
        l_temp = np.ndarray((u.shape[0], 2), dtype=DataStruct)
        # (x2) U0 | U1 | ...
        for i in range(4):
            n_diag=16*i
            delta = (((64-n_diag)%64)*16)%(2**15)
            rot_i = self.engine.rotate_left(u[i], delta, self.engine.rot_keys[delta])            
            mask0 = self.masks['transpose']['mask0'][i]
            mask1 = self.masks['transpose']['mask1'][i]
            temp0 = self.engine.rescale(self.engine.pt_ct_mult(mask0, rot_i))
            temp1 = self.engine.rescale(self.engine.pt_ct_mult(mask1, rot_i))
            #0,3,2,1
            l_temp[(4-i)%4][0] = temp0
            l_temp[(4-i)%4][1] = temp1
            
        for i in range(4):    
            for n_diagU in range(16*i+1,16*(i+1)):
                l = 64 - n_diagU
                delta = (l*2**4 + (((n_diagU%48)*2)%16)*2**11)%(2**15)
                rot_i_1 = self.engine.rotate_left(u[i], delta, self.engine.rot_keys[delta])  
                
                mask2 = self.masks['transpose']['mask2'][n_diagU]
                mask3 = self.masks['transpose']['mask3'][n_diagU]
                
                temp2 = self.engine.rescale(self.engine.pt_ct_mult(mask2, rot_i_1))
                temp3 = self.engine.rescale(self.engine.pt_ct_mult(mask3, rot_i_1))
                
                #3,2,1,0
                l_temp[(3-i)%4][0] = self.engine.add(l_temp[(3-i)%4][0], temp2)
                l_temp[(3-i)%4][1] = self.engine.add(l_temp[(3-i)%4][1], temp3)
        
        l = np.full((4,), None, dtype=DataStruct)
        for i in range(4):
            l[i] = self.engine.add(l_temp[i][0], self.engine.rotate_left(l_temp[i][1], -2**11))
        return l
    
    def make_rotated_copies(self, cts: np.ndarray[DataStruct]):
        """
        Input cts: [L0, L1, ..., L15], [L16, L17, ..., L31], ...(ct array of shape (n,))
        Ouptut cts: [L0, L1, ..., L15], [L1, L2, ..., L0], ...(ct array of shape (16*n,))
        """
        rots = np.full((16*cts.shape[0],), None, dtype=DataStruct)
        for i in range(cts.shape[0]):
            rots[16*i] = cts[i]
            for j in range(1, 16):
                rots[16*i+j] = self.engine.rotate_left(rots[16*i+j-1], 2**11)
        return rots
    
    def make_copies(self, cts: np.ndarray[DataStruct],scale=1/2):
        """
        Input cts: [L0, L1, ..., L15], [L16, L17, ..., L31], ...(ct array of shape (n,))
        Ouptut cts: scale * [L0, L0, ... L0], [L1, L1, ... L1], ...(ct array of shape (16*n,)), scale is 1/2 by default
        """
        copies = np.full((16*cts.shape[0],), None, dtype=DataStruct)
        scale = scale/2
        masks = self.masks['make_copies_2']
        
        mask0 = np.full((2**15,), 1)
        mask0[np.arange(2**15) % (2**12) >= 2**11] = 0
        mask1 = np.full((2**15,), 1)
        mask1[np.arange(2**15) % (2**12) < 2**11] = 0

        for i in range(cts.shape[0]//2):
            ct0 = cts[i]
            ct1 = cts[i+cts.shape[0]//2]
            merged_ct = self.engine.cc_add(ct0, self.engine.imult(ct1))
            for j in range(8):
                masked_ct = self.engine.rescale(self.engine.pt_ct_mult(masks[j],merged_ct))
                copied_ct= self.rotsum(masked_ct, 2**12)
                copied_ct_0 = self.engine.mc_mult(mask0, copied_ct)
                copied_ct_1 = self.engine.mc_mult(mask1, copied_ct)
                
                copied_ct_0 = self.engine.cc_add(copied_ct_0, self.engine.rotate_left(copied_ct_0, -2**11))
                copied_ct_1 = self.engine.cc_add(copied_ct_1, self.engine.rotate_left(copied_ct_1, 2**11))

                conj_0 = self.engine.conjugate(copied_ct_0)
                conj_1 = self.engine.conjugate(copied_ct_1)

                copies[i*16+2*j]= self.engine.cc_add(copied_ct_0, conj_0)
                copies[(i+cts.shape[0]//2)*16+2*j]= self.engine.imult(self.engine.cc_sub(conj_0, copied_ct_0))
                
                copies[i*16+2*j+1]= self.engine.cc_add(copied_ct_1, conj_1)
                copies[(i+cts.shape[0]//2)*16+2*j+1]= self.engine.imult(self.engine.cc_sub(conj_1, copied_ct_1))
                
        return copies #Level: l + 1

    def rotate_internal(self, ct:DataStruct, delta:int=0, l_delta = 0, r_delta = 0, mask=None, mode=None) -> DataStruct:
        """
        Rotates the ciphertext internally.
        @param mode: 'att' or 'block_diag_1' or 'block_diag_2'
        'block_diag_1' for qkv(linear projection) and attention_output
        'block_diag_2' for feed forward layer
        """
        if delta == 0 and l_delta ==0:
            return self.engine.level_up(ct, ct.level_calc+1)
        if mask is None:
            try:
                mask = self.masks['rot_internal'][mode][delta]
            except Exception as e:
                raise ValueError(f"Internal rotation mask for {mode}, {delta} is not precomputed")
        if mode == 'att':
            l_delta = 16*delta
            r_delta = 2**11 - l_delta
        elif mode == 'block_diag_1':
            l_delta = delta
            r_delta = 12 - l_delta
        elif mode == 'block_diag_2':
            l_delta = delta
            r_delta = 6 - l_delta
        temp1 = self.engine.rescale(self.engine.pt_ct_mult(mask, ct)) #For right rotate level: l -> l+1
        temp2 = self.engine.cc_sub(self.engine.level_up(ct, temp1.level_calc), temp1) #For left rotate level: l -> l+1
        rrot_ct = self.engine.rotate_left(temp1, -r_delta)
        lrot_ct = self.engine.rotate_left(temp2, l_delta)
        return self.engine.add(lrot_ct, rrot_ct)

    def rotsum(self, ct:DataStruct, interval:int) -> DataStruct:
        """
        Rotate Sum Operation
        """
        rep = int(np.log2(self.engine.num_slots/interval))
        temp = ct
        for i in range(rep):
            temp = self.engine.cc_add(temp, self.engine.rotate_left(temp, interval*2**i))
        return temp
    
    def pre_encode_masks(self):
        """
        Pre encode masks for algorithms such as internal rotations, make_copies and transpose.
        """
        self.masks['rot_internal'] = {'att':{}, 'block_diag_1':{}, 'block_diag_2':{}}
        self.masks['make_copies'] = {}
        self.masks['make_copies_merge'] = {}
        self.masks['make_copies_2'] = {}
        self.masks['transpose'] = {'mask0':{}, 'mask1':{}, 'mask2':{}, 'mask3':{}}
        self.masks['ct_ct_matmul'] = {0:{}, 1:{}, 2:{}, 3:{}}
        level = 15
        
        for i in range(1, 128):
            array = np.ones((2**15,), dtype=int)
            array[np.arange(self.engine.num_slots) % (2**11) >= (16*i)] = 0
            self.masks['rot_internal']['att'][i] = self.engine.encode(array, level= level)

        for i in range(1, 16):
            array = np.ones((self.engine.num_slots,), dtype=int)
            array[np.arange(self.engine.num_slots) % (16) >= i] = 0
            self.masks['rot_internal']['block_diag_1'][i] = self.engine.encode(array, level= level)
        
        for i in range(1, 8):
            array = np.ones((self.engine.num_slots,), dtype=int)
            array[np.arange(self.engine.num_slots) % (8) >= i] = 0
            self.masks['rot_internal']['block_diag_2'][i] = self.engine.encode(array, level= level)
        
        for i in range(16):
            arr = np.zeros((2**15,), dtype=int)
            arr[2**11*i:2**11*(i+1)] = np.full((2**11,), 1)
            self.masks['make_copies_merge'][i] = self.engine.encode(arr*(1/4), level=level)
            
        for i in range(8):
            arr = np.zeros((2**15,), dtype=int)
            arr[2**12*i:2**12*(i+1)] = np.full((2**12,), 1)
            self.masks['make_copies_2'][i] = self.engine.encode(arr*(1/4), level=level)
            
        for i in range(4):
            n_diag = 16*i
            arr0 = np.array([1]*16*(64+(n_diag-16)%64+16)) 
            arr1 = np.array([0]*(2**15-16*(64-((n_diag-16)%64+16)))+[1]*16*(64-((n_diag-16)%64+16))) 
            self.masks['transpose']['mask0'][i] = self.engine.encode(arr0, level=level)
            self.masks['transpose']['mask1'][i] = self.engine.encode(arr1, level=level)
            for j in range(16*i+1, 16*(i+1)):
                l = 64-j
                arr2 = np.array([0]*2**11*(16-j%16)+[1]*(128-l)*2**4) 
                arr3 = np.array([0]*2**11*(16-j%16-1)+[0]*(128-l)*2**4+[1]*16*l) 
                self.masks['transpose']['mask2'][j] = self.engine.encode(arr2, level=level)
                self.masks['transpose']['mask3'][j] = self.engine.encode(arr3, level=level)
        
        for n in range(1, 128):
            masks = self.masks['ct_ct_matmul']
            rot = n
            j = n % 16
            arr0 = np.full((2**15,), 1, dtype=float)
            arr0[np.arange(2**15) % (2**11) >= (2**11 - 16*rot)] = 0
                    
            arr1 = np.full((2**15,), 0, dtype=float)
            arr1[np.arange(2**15) % (2**11) >= (2**11 - 16*rot)] = 1
            
            if j == 0:
                masks[0][n] = self.engine.encode(arr0, level=level)
                masks[1][n] = self.engine.encode(arr1, level=level)
            else:
            
                arr0[:(2**11)*j] = np.zeros(((2**11)*j,), dtype=float)
                masks[0][n] = self.engine.encode(arr0, level=level)
        
                arr1[-(2**11):] = np.zeros(((2**11)), dtype=float)
                if j > 1:
                    arr1[:(2**11)*(j-1)] = np.zeros(((2**11)*(j-1)), dtype=float)
                masks[1][n] = self.engine.encode(arr1, level=level)
                
                arr2 = np.full((2**15,), 1, dtype=float)
                arr2[np.arange(2**15) % (2**11) >= (2**11 - 16*rot)] = 0
                arr2[(2**11)*j:] = np.zeros(((2**11)*(16-j)), dtype=float)
                masks[2][n] = self.engine.encode(arr2, level=level)
                
                arr3 = np.full((2**15,), 1, dtype=float)
                arr3 = arr3 - arr0 - arr1 - arr2
                masks[3][n] = self.engine.encode(arr3, level=level)
                

