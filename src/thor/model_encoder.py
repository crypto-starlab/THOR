import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from liberate.fhe.data_struct import DataStruct

from .ckks import CkksEngine
from .utils.matrix import ud_entry, to_blocks

class ThorModelEncoder:
    def __init__(self, engine:CkksEngine, model_dir: str):
        self.ckks_engine = engine
        self.weights_m:dict[str, torch.Tensor] = load_safetensors(model_dir)  #Params(Message)
        self.weights_pt:dict[str, torch.Tensor] = {key: None for key in self.weights_m.keys()} #Params(Plaintext)
        self.n_layers = 1 #Change later. For testing purpose
        self.n_heads = 12
        self.pad_index = {'att': (12, 13, 14, 15), 'ff': (6,7,14,15)}
    
    def encode_model(self):
        """
        Encode the model weights into plaintexts array of dimension (ll, n_in, n_out/pack) or (2, ll, n_in, n_out/pack) for ff
        QKV: Array of shape (6, 64, 4), total 1536 plaintexts
        AttOutput: Array of shape (6, 64, 8), total 3072 plaintexts
        FF1: Array of shape (2, 6, 64, 8), total 6144 plaintexts
        FF2: Array of shape (2, 6, 64, 8), total 6144 plaintexts
        
        """
        for layer in range(self.n_layers):
            self.encode_att(layer)
            self.encode_ff(layer)
            self.encode_pooler()
            self.encode_cls()
            
    def encode_att(self, layer:int):
        for qkv in ['query', 'key', 'value']:
            if qkv == 'key':
                sftmx_scale = 1/512 if layer != 2 else 1/1024
            else: 
                sftmx_scale = 1
            self.weights_pt[f'bert.encoder.layer.{layer}.attention.self.{qkv}.weight'] = self._encode_w_qkv(
                self.weights_m[f'bert.encoder.layer.{layer}.attention.self.{qkv}.weight'].cpu().numpy(),level=21, scale=sftmx_scale
                )
            self.weights_pt[f'bert.encoder.layer.{layer}.attention.self.{qkv}.bias'] = self._encode_b(
                self.weights_m[f'bert.encoder.layer.{layer}.attention.self.{qkv}.bias'].cpu().numpy(),level=22,
                n_blocks = self.n_heads, n_out = 64, scale=sftmx_scale
                )
        
        #Change level later
        self.weights_pt[f'bert.encoder.layer.{layer}.attention.output.dense.weight'] = self._encode_w_att(
            self.weights_m[f'bert.encoder.layer.{layer}.attention.output.dense.weight'].cpu().numpy(), level=18,
            n_in = 64, n_out = 128, b_shape=(128, 64)
            )
        self.weights_pt[f'bert.encoder.layer.{layer}.attention.output.dense.bias'] = self._encode_b(
            self.weights_m[f'bert.encoder.layer.{layer}.attention.output.dense.bias'].cpu().numpy(), level=20,
            n_blocks = 6, n_out = 128
            )
        self.weights_pt[f'bert.encoder.layer.{layer}.attention.output.LayerNorm.weight'] = self._encode_b(
            self.weights_m[f'bert.encoder.layer.{layer}.attention.output.LayerNorm.weight'].cpu().numpy(), 
            n_blocks = 6, n_out = 128, level=15
            )
        #This level should be changed
        self.weights_pt[f'bert.encoder.layer.{layer}.attention.output.LayerNorm.bias'] = self._encode_b(
            self.weights_m[f'bert.encoder.layer.{layer}.attention.output.LayerNorm.bias'].cpu().numpy(), 
            n_blocks = 6, n_out = 128, level=15
            )
    
    def encode_ff(self, layer:int):
        gelu_scale = 1/64
        for ff_type in {'intermediate', 'output'}:
            if ff_type == 'intermediate':
                weight = self.weights_m[f'bert.encoder.layer.{layer}.{ff_type}.dense.weight'].cpu().numpy()
                if weight.shape != (3072, 768):
                    raise ValueError("Shape of FF1 W should be (3072, 768)")

                self.weights_pt[f'bert.encoder.layer.{layer}.{ff_type}.dense.weight'] = self._encode_w_ff(
                    weight, n_in = 128, n_out= 128, b_shape=(128, 128),vsplit=4, scale=gelu_scale, level=16
                    )
    
                self.weights_pt[f'bert.encoder.layer.{layer}.{ff_type}.dense.bias'] = np.full((2,8), None, dtype=DataStruct)
                
                bias_m = np.split(self.weights_m[f'bert.encoder.layer.{layer}.{ff_type}.dense.bias'].cpu().numpy(), 2)
                
                for i in range(2):
                    self.weights_pt[f'bert.encoder.layer.{layer}.{ff_type}.dense.bias'][i] = self._encode_b(
                    bias_m[i], n_blocks = 12, n_out = 128, pad_index=self.pad_index['ff'], scale=gelu_scale, level=18
                    )
            else:
                weight = self.weights_m[f'bert.encoder.layer.{layer}.{ff_type}.dense.weight'].cpu().numpy()
                if weight.shape != (768, 3072):
                    raise ValueError("Shape of FF2 W should be (768, 3072)")
                
                self.weights_pt[f'bert.encoder.layer.{layer}.{ff_type}.dense.weight'] = self._encode_w_ff(
                    weight, n_in = 128, n_out= 128, b_shape=(128, 128), hsplit=4, level=18
                    )
                
                self.weights_pt[f'bert.encoder.layer.{layer}.{ff_type}.dense.bias'] = self._encode_b(
                self.weights_m[f'bert.encoder.layer.{layer}.{ff_type}.dense.bias'].cpu().numpy(), 
                n_blocks = 6, n_out = 128, n_slot=16, level=20
                )
                
        self.weights_pt[f'bert.encoder.layer.{layer}.output.LayerNorm.weight'] = self._encode_b(
        self.weights_m[f'bert.encoder.layer.{layer}.output.LayerNorm.weight'].cpu().numpy(),
        n_blocks = 6, n_out = 128, n_slot=16, level=15
        )
        
        self.weights_pt[f'bert.encoder.layer.{layer}.output.LayerNorm.bias'] = self._encode_b(
            self.weights_m[f'bert.encoder.layer.{layer}.output.LayerNorm.bias'].cpu().numpy(),
        n_blocks = 6, n_out = 128, n_slot=16, level=15
        )
                
    def encode_pooler(self, level:int=15):
        self.weights_pt['bert.pooler.dense.weight'] = self._encode_w_pooler(
            self.weights_m['bert.pooler.dense.weight'].cpu().numpy()
            )
        self.weights_pt['bert.pooler.dense.bias'] = self._encode_b_pooler(
            self.weights_m['bert.pooler.dense.bias'].cpu().numpy(), n_blocks = 6
            )
        
    def encode_cls(self, level:int=15):
        
        cls_name = "cls.seq_relationship" if 'cls.seq_relationship.weight' in self.weights_m.keys() else "classifier"
        w = self.weights_m[f'{cls_name}.weight'].cpu().numpy()
        b = self.weights_m[f'{cls_name}.bias'].cpu().numpy()
        
        self.weights_pt[f'{cls_name}.weight'] = self._encode_w_cls(
            w
            )
        
        self.weights_pt[f'{cls_name}.bias'] = self._encode_b_cls(
            b
            )
        
    def _encode_w_qkv(self, w:np.ndarray, level:int=15, scale=1/256) -> np.ndarray:
        """
        Return an array of shape (4, 6, 64) which contains 4 * 6 * 64 = 1536 plaintexts
        """
        if w.shape != (768, 768):
            raise ValueError("Shape of Wq, Wk, Wv matrices should be (768, 768)")
        w_pt = self._encode_w_att(w, n_in = 128, n_out= 64, b_shape=(64, 128), level=level, scale=scale)
        return w_pt
    
    def _encode_w_att(self, w:np.ndarray, n_in:int, n_out:int, b_shape:tuple[int], level:int=15, scale:float=1) -> np.ndarray:
        """
        Returns plaintext array of shape (n_out/pack, ll, n_in/2), ll = min(w.shape[0]/b_shape[0], w.shape[1]/b_shape[1])
        """
        pack = 16
        dim = 128
        
        if w.shape[0] % b_shape[0] != 0 or w.shape[1] % b_shape[1] != 0:
            raise ValueError("Dimension does not match")
        if n_out % pack != 0:
            raise ValueError("Number of n_out should be divisible by pack")

        n_in_c = n_in //2 #Complexification
        ld_blocks, (ll,dd) = to_blocks(w, b_shape, diag=True) 
        n_out_p = n_out // pack
        pts = np.full((n_out_p, ll, n_in_c), None, dtype=DataStruct)

        for l in range(ll):
            diagonal = ld_blocks[l]
            for n in range(n_in_c): 
                for out in range(n_out_p): #Comput r0~r16, r16~r32, r32~r48, r48~r64
                    msg = np.zeros((2**15,), dtype=complex)
                    for j in range(pack):
                        i = ((n // 16) * 16 + out *16+ (n+j)% 16) % n_in_c  # i = 0, 1, ..., 63
                        r = out *16 + j # L_out : rot = 0, 1, ..., 63
                        i = (i - r) % n_in
                        temp = j* (2**11)
                        for t in range(dim):
                            for d in range(12): 
                                block = diagonal[d]
                                msg[temp + t*16 + d] = complex( (scale*ud_entry(block, i, t, r))/2, - (scale*ud_entry(block, (i+n_in_c)%n_in, t, r))/2)
                    pt = self.ckks_engine.encode(msg, level)
                    pt = pt[0].cpu()
                    torch.cuda.empty_cache()
                    pts[out, l, n] = pt
        return pts
        
    def _encode_w_ff(self, w:np.ndarray, n_in:int, n_out:int, b_shape:tuple[int], 
                         level:int=15, vsplit =0, hsplit=0, scale=1) -> np.ndarray:
        """
        Returns plaintext array of shape (2, n_out/pack, ll, n_in/2), ll = min(w.shape[0]/b_shape[0], w.shape[1]/b_shape[1])
        """
        pack = 16
        dim = 128
        n_in_c = n_in //2 #Complexification
        if w.shape[0] % b_shape[0] != 0 or w.shape[1] % b_shape[1] != 0:
            raise ValueError("Dimension does not match")
        if n_out % pack != 0:
            raise ValueError("Number of n_out should be divisible by pack")

        if vsplit:
            w_list = np.vsplit(w, vsplit)
            
        if hsplit:
            w_list = np.hsplit(w, hsplit)
            
        ld_blocks_list = [] #split * (ll, dd) = 4 * (6,6)
        
        for w in w_list:
            ld_blocks, (ll,dd) = to_blocks(w, b_shape, diag=True) #ll = 6, dd = 6
            ld_blocks_list.append(ld_blocks)
            
        n_out_packed = n_out // pack

        pts = np.full((2, n_out_packed, ll, n_in_c), None, dtype=object) #(2, 8, 6, 64)
        for rep in range(2):
            for l in range(ll):
                for n in range(n_in_c): 
                    for out in range(n_out_packed): #Comput r0~r16, r16~r32, ...r112~r127
                        msg = np.zeros((2**15,), dtype=complex)
                        for j in range(pack):
                            i = ((n // 16) * 16 + out *16+ (n+j)% 16) % n_in_c  # i = 0, 1, ..., 63
                            r = out *16 + j # L_out : rot = 0, 1, ..., 63
                            i = (i - r) % n_in
                            temp = j* (2**15//pack)
                            for t in range(dim):
                                for d in range(6): 
                                    block1 = ld_blocks_list[rep * 2 + 0][l, d]
                                    block2 = ld_blocks_list[rep * 2 + 1][l, d]
                                    msg[temp + t*16 + d] = complex((scale*ud_entry(block1, i, t, r)/2), -((scale*ud_entry(block1, (i+n_in_c)%n_in, t, r))/2))
                                    msg[temp + t*16 + d+8] = complex((scale*ud_entry(block2, i, t, r)/2), -((scale*ud_entry(block2, (i+n_in_c)%n_in, t, r))/2))
                        pt = self.ckks_engine.encode(msg, level)
                        pt = pt[0].cpu()
                        with torch.cuda.device(1):
                            torch.cuda.empty_cache()
                        pts[rep, out, l, n]= pt
        return pts
    
    def _encode_w_pooler(self, w:np.ndarray, b_shape:tuple[int]=(128,128), level:int=15) -> np.ndarray:
        """
        Returns plaintext array of shape (ll, 4), ll = min(w.shape[0]/b_shape[0], w.shape[1]/b_shape[1])
        """
        pack = 16
        
        if w.shape[0] % b_shape[0] != 0 or w.shape[1] % b_shape[1] != 0:
            raise ValueError("Dimension does not match")

        ld_blocks, (ll,dd) = to_blocks(w, b_shape, diag=True) #ll = 6, dd = 6
        pts = np.full((ll, 4), None, dtype=DataStruct)
        for l in range(ll):
            blocks = ld_blocks[l]
            for n in range(4): 
                msg = np.zeros((2**15,), dtype=complex)
                for j in range(pack):
                    i = n *16 + j  # i = 0, 1, ..., 63
                    temp = j* (2**11)
                    for m in range(2**7):
                        for d in range(6): 
                            block = blocks[d]
                            msg[temp + m * 16 + d] = complex(block[m, i]/2, -block[m, i+64]/2)
                pt = self.ckks_engine.encode(msg, level)
                pt = pt[0].cpu()
                torch.cuda.empty_cache()
                pts[l, n] = pt
        return pts
            
    def _encode_w_cls(self, w:np.ndarray, level:int=15) -> np.ndarray:
        """
        @w: numpy array of shape (cls, 768)
        """
        if w.shape[1] != 768:
            raise ValueError(f"Shape of W_cls should be (cls, 768). Shape of W_cls is {w.shape}")
        n_cls = w.shape[0]
        pts = np.full((n_cls,), None, dtype=DataStruct)
        for n in range(n_cls):
            blocks = np.split(w[n], 6)
            msg = np.zeros((2**15,), dtype=float)
            for t in range(128):
                for d in range(6):
                    block = blocks[d]
                    msg[t* 16 + d] = block[t]
            pt = self.ckks_engine.encode(msg, level)
            pt = pt[0].cpu()
            torch.cuda.empty_cache()
            pts[n] = pt
        return pts
    
    def _encode_b_cls(self, b:np.ndarray, level:int=15) -> np.ndarray:
        n_cls = b.shape[0]
        msg = np.zeros((2**15,), dtype=float)
        pts = np.full((n_cls,), None, dtype=DataStruct)
        for n in range(n_cls):
            msg = np.zeros((2**15,), dtype=float)
            msg[0] = b[n]
            pt = self.ckks_engine.encode(msg, level)
            pt = pt[0].cpu()
            pts[n] = pt
        return pts
    
    def _encode_b(self, b:np.ndarray, n_blocks:int, n_out:int, 
                  level:int =15,pack:int=16, n_slot = 16, pad_index=None, scale=1) -> np.ndarray:
        """
        Returns an array of shape (n_out/pack, ) which contains n_out/pack plaintexts
        """
        dim = 128
        if pad_index is None:
            pad_index = [i for i in range(n_slot) if i >= n_blocks]
        else:
            if n_blocks + len(pad_index) != n_slot:
                raise ValueError("Parameters do not match")
        if type(b) != np.ndarray:
            raise ValueError("Input should be a numpy array")
        if b.shape[0] % n_blocks != 0:
            raise ValueError("Block size does not match")
        blocks = np.split(b, n_blocks) #bias for Q_0, Q_1, ..., Q_11
        n_out_packed = n_out // pack
        pts = np.full((n_out_packed,),None, dtype=object)
        for out in range(n_out_packed):
            msg = np.zeros((2**15,), dtype=float)
            for j in range(pack):
                temp = j * (2**11)
                r = out * pack + j #0 ~ 63
                for t in range(dim):
                    c = 0
                    for d in range(n_slot):
                        if d in pad_index:
                            pass
                        else: 
                            block = blocks[c]
                            msg[temp + t*n_slot + d] = (scale * block[(r+t) % block.shape[0]])/2
                            c += 1
            pt = self.ckks_engine.encode(msg, level)
            pt = pt[0].cpu()
            torch.cuda.empty_cache()
            pts[out] = pt
        return pts
        
    def _encode_b_pooler(self, b:np.ndarray, n_blocks:int, 
                  level:int =15,n_slot = 16, pad_index=None) -> np.ndarray:
        """
        Returns an array of shape (n_out/pack, ) which contains n_out/pack plaintexts
        """
        dim = 128
        if pad_index is None:
            pad_index = [i for i in range(n_slot) if i >= n_blocks]
        else:
            if n_blocks + len(pad_index) != n_slot:
                raise ValueError("Parameters do not match")
        if type(b) != np.ndarray:
            raise ValueError("Input should be a numpy array")
        if b.shape[0] % n_blocks != 0:
            raise ValueError("Block size does not match")
        blocks = np.split(b, n_blocks) #bias for Q_0, Q_1, ..., Q_11
        pts = np.full((1,),None, dtype=object)
        msg = np.zeros((2**11,), dtype=float)
        for t in range(dim):
            c = 0
            for d in range(n_slot):
                if d in pad_index:
                    pass
                else: 
                    block = blocks[c]
                    msg[t*n_slot + d] = block[t]/2
                    c += 1
        msg = np.tile(msg, 2**4)
        pt = self.ckks_engine.encode(msg, level)
        pt = pt[0].cpu()
        torch.cuda.empty_cache()
        pts[0] = pt
        return pts