import torch 
import numpy as np
from liberate.fhe.data_struct import DataStruct
from liberate.fhe.presets import types, errors
from liberate.fhe import ckks_engine
from liberate.fhe.encdec import rotate
from liberate.fhe.bootstrapping import ckks_bootstrapping as bs

print(ckks_engine)

class CkksEngine(ckks_engine):
    def __init__(self, params, verbose=False):
        """
        CkksEngine is a child class of Liberate ckks_engine for basic CKKS operations.
        """
        super().__init__(**params, verbose=verbose)
        self.rot_keys = np.full((2**15,), None, dtype=DataStruct) #Note that this is right rotation, as Liberate uses right rotation as default
        self.hrot_keys = np.full((2**15,), None, dtype=DataStruct)
        self.pk = None
        self.evk = None
        self.conj_key = None
        self.fft_error = None
        self.one = None
    
    #Precomputation
    def add_pk(self, pk:DataStruct):
        self.pk = pk
        
    def add_rot_keys_from_sk(self, deltas:list[int], sk:DataStruct):
        """
        Add left rotation keys for the given deltas.
        """
        for delta in deltas:
            if delta < 0:
                delta = self.num_slots + delta
            if self.rot_keys[delta] == None:
                self.rot_keys[delta] = self.create_rotation_key(sk, self.num_slots - delta)
            
    def add_hrot_keys_from_sk(self, deltas:list[int], sk:DataStruct):
        """
        Add left hoist rotation keys for the given deltas.
        """
        for delta in deltas:
            if delta < 0:
                delta = self.num_slots + delta
            self.hrot_keys[delta] = self.create_hoisting_rotation_key(sk, self.num_slots - delta)
            
    def add_bs_key(self, bsk:DataStruct):
        self.bs_key = bsk
            
    def add_evk(self, evk:DataStruct):
        self.evk = evk

    def add_gk(self, gk:DataStruct):
        self.gk = gk
        
    def add_conj_key(self, conj_key:DataStruct):
        self.conj_key = conj_key
            
    #Basic Operations
    def encode_and_encrypt(self, m, pk=None, level: int = 0, padding=True) -> DataStruct:
        if pk is None:
            if self.pk is None:
                raise ValueError("Public key is not set")
            pk = self.pk
        return super().encodecrypt(m, pk, level, padding)
    
    def bootstrap(self, ct):
        for device in self.ntt.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        temp = ct
        ct_bs = bs.bootstrap(self, temp, self.bs_key, self.evk, self.conj_key, self.pk)
        for device in self.ntt.devices:
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
        return ct_bs
    
    #Mult
    def mult(self, a, b, evk=None, relin:bool=True, auto_level=False) -> DataStruct:
        if evk is None:
            if self.evk is None:
                raise ValueError("Evaluation key is not set")
            evk = self.evk
        return super().mult(a, b, evk, relin=relin)
    
    def auto_ct_ct_mult(self, ct0: DataStruct, ct1: DataStruct, evk=None, relin=True, rescale=True) -> DataStruct:
        if evk is None:
            if self.evk is None:
                raise ValueError("Evaluation key is not set")
            evk = self.evk
        lct0, lct1 = self.auto_level(ct0, ct1)
        return self.ct_ct_mult(lct0, lct1, evk, relin=relin, rescale=rescale)

    def ct_ct_mult(self, a: DataStruct, b: DataStruct, evk =None, relin=True, rescale=True) -> DataStruct:
        return super().cc_mult(a, b, evk, relin, rescale)
    
    def relinearize(self, ct_triplet: DataStruct, evk: DataStruct=None, is_fast:bool=True) -> DataStruct:
        if evk is None:
            if self.evk is None:
                raise ValueError("Evaluation key is not set")
            evk = self.evk
        return super().relinearize(ct_triplet=ct_triplet, evk=evk, is_fast=is_fast)
    
    def sqrt(self, ct: DataStruct, evk=None, e=0.0001, alpha=0.0001) -> DataStruct:
        if evk is None:
            if self.evk is None:
                raise ValueError("Evaluation key is not set")
            evk = self.evk
        return super().sqrt(ct, self.evk, e, alpha)
    
    def square(self, ct: DataStruct, evk =None, relin=True, is_fast: bool = True):
        if evk is None:
            if self.evk is None:
                raise ValueError("Evaluation key is not set")
            evk = self.evk
        return super().square(ct, evk, relin, is_fast)
    
    def imult(self, x: np.ndarray[DataStruct]) -> np.ndarray[DataStruct]:
        return self.mult_imag(x)
    
    def minus_imult(self, x: np.ndarray[DataStruct]) -> np.ndarray[DataStruct]:
        return self.mult_imag(x, True)
    
    def pt_ct_mult(self, pt, ct) -> DataStruct:
        """
        Plaintext - Ciphertext multiplication.
        """
        
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchmode(origin=ct.origin, to=types.origins["sk"])
            
        pt_tiled = self.ntt.tile_unsigned(pt, ct.level_calc)
        # Transform ntt to prepare for multiplication.
        self.ntt.enter_ntt(pt_tiled, ct.level_calc)
        # Prepare a new ct.
        new_ct = self.clone(ct)


        self.ntt.enter_ntt(new_ct.data[0], ct.level_calc)
        self.ntt.enter_ntt(new_ct.data[1], ct.level_calc)

        new_d0 = self.ntt.mont_mult(pt_tiled, new_ct.data[0], ct.level_calc)
        new_d1 = self.ntt.mont_mult(pt_tiled, new_ct.data[1], ct.level_calc)
        
        self.ntt.intt_exit_reduce(new_d0, ct.level_calc)
        self.ntt.intt_exit_reduce(new_d1, ct.level_calc)

        new_ct.data[0] = new_d0
        new_ct.data[1] = new_d1

        return new_ct
    
    def pt_ct_mult_extended(engine, pt, ct):
        pt_tiled = engine.ntt.tile_unsigned(pt, ct.level_calc)
        engine.ntt.enter_ntt(pt_tiled, ct.level_calc)
        new_ct = engine.clone(ct)
        new_d0 = engine.ntt.mont_mult(pt_tiled, new_ct.data[0], ct.level_calc)
        new_d1 = engine.ntt.mont_mult(pt_tiled, new_ct.data[1], ct.level_calc)
        new_d2 = engine.ntt.mont_mult(pt_tiled, new_ct.data[2], ct.level_calc)
        new_ct.data[0] = new_d0
        new_ct.data[1] = new_d1
        new_ct.data[2] = new_d2
        return new_ct
    
    def mult_int_scalar_triplet(self, ct, scalar, evk=None, relin=True):
        device_len = len(ct.data[0])
        data_len = len(ct.data)
        level = ct.level_calc

        int_scalar = int(scalar)
        mont_scalar = [(int_scalar * self.ctx.R) % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[level]

        partitioned_mont_scalar = [[mont_scalar[i] for i in desti] for desti in dest]
        tensorized_scalar = []
        for device_id in range(device_len):
            scal_tensor = torch.tensor(
                partitioned_mont_scalar[device_id],
                dtype=self.ctx.torch_dtype,
                device=self.ntt.devices[device_id]
            )
            tensorized_scalar.append(scal_tensor)

        new_ct = self.clone(ct)
        new_data = new_ct.data

        for i in range(data_len):          
            self.ntt.mont_enter_scalar(new_data[i], tensorized_scalar, level)
            self.ntt.reduce_2q(new_data[i], level)

        return new_ct

    def pc_add(self, pt, ct) -> DataStruct:
        """
        Note that if the level of pt and ct is different, the result will be incorrect.
        """
        pt_tiled = self.ntt.tile_unsigned(pt, ct.level_calc)

        self.ntt.mont_enter_scale(pt_tiled, ct.level_calc)

        new_ct = self.clone(ct)
        self.ntt.mont_enter(new_ct.data[0], ct.level_calc)
        new_d0 = self.ntt.mont_add(pt_tiled, new_ct.data[0], ct.level_calc)
        self.ntt.mont_redc(new_d0, ct.level_calc)
        self.ntt.reduce_2q(new_d0, ct.level_calc)

        new_ct.data[0] = new_d0

        return new_ct
    
    #Rotations
    def rotate_left(self, ct:DataStruct, delta:int=2**60, rot_key=None) -> DataStruct:
        if delta == 0:
            return ct
        elif delta < 0:
            delta = self.num_slots + delta
        if rot_key is None:
            if delta == 2**60:
                raise ValueError("Either delta or rot_key should be given")
            elif self.rot_keys[delta] is None:
                return self.rotate_galois(ct, self.gk, -delta)
            else:
                rot_key = self.rot_keys[delta]
        return self.rotate_single(ct, rot_key)
    
    def conjugate(self, ct:DataStruct, conj_key=None) -> DataStruct:
        if conj_key is None:
            if self.conj_key is None:
                raise ValueError("Conjugation key is not set")
            conj_key = self.conj_key
        return super().conjugate(ct, conj_key)    

    def rotate_hoist(self, ct: DataStruct, delta: DataStruct, rotk=None) -> DataStruct:
        if rotk is None:
            rotk = self.hrot_keys[delta]
        include_special = ct.include_special
        ntt_state = ct.ntt_state
        montgomery_state = ct.montgomery_state
        origin = rotk.origin
        delta = int(origin.split(':')[-1])
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        level = ct.level_calc
        ct_clone = self.clone(ct)
        c0 = ct_clone.data[0]
        c1 = ct_clone.data[1]
        decomposed_part_results = self.decompose(c1, level, exit_ntt=False)
        self.ntt.mont_enter_scalar(c0, [self.mont_PR[i][-len(self.ntt.p.destination_arrays[level][i]):] for i in
                                        range(self.ntt.num_devices)], lvl=level, mult_type=-1)
        self.ntt.mont_enter_scalar(c1, [self.mont_PR[i][-len(self.ntt.p.destination_arrays[level][i]):] for i in
                                        range(self.ntt.num_devices)], lvl=level, mult_type=-1)
        c0 = [torch.cat([c0[device_id],
                            torch.zeros([self.ntt.num_special_primes, self.ctx.N], device=self.ntt.devices[device_id],
                                        dtype=self.ctx.torch_dtype)], dim=0) for device_id in range(self.ntt.num_devices)]
        summed0, summed1 = self.mult_sum(decomposed_part_results, rotk, level)
        self.ntt.intt_exit_reduce(summed0, level, mult_type=-2)
        self.ntt.intt_exit_reduce(summed1, level, mult_type=-2)
        summed0 = self.ntt.mont_add(c0, summed0, level, mult_type=-2)
        summed0 = [rotate(d, delta) for d in summed0]
        summed1 = [rotate(d, delta) for d in summed1]
        self.ntt.make_unsigned(summed0, level, mult_type=-2)
        self.ntt.reduce_2q(summed0, level, mult_type=-2)
        self.ntt.make_unsigned(summed1, level, mult_type=-2)
        self.ntt.reduce_2q(summed1, level, mult_type=-2)
        d0 = self.mod_down(summed0, level)
        d1 = self.mod_down(summed1, level)
        return DataStruct(
            data=(d0, d1),
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            origin=types.origins["ct"],
            level=level,
            hash=self.hash
        )
    
    def rotsum(self, ct:DataStruct, interval:int) -> DataStruct:
        """
        Rotate Sum Operation
        """
        rep = int(np.log2(self.num_slots/interval))
        temp = ct
        for i in range(rep):
            temp = self.cc_add(temp, self.rotate_left(temp, interval*2**i))
        return temp
    
    def cc_add(self, a: DataStruct, b: DataStruct) -> DataStruct:
        if a is None:
            return b
        elif b is None:
            return a
        if a.origin == types.origins["ct"] and b.origin == types.origins["ct"]:
            ct_add = self.cc_add_double(a, b)
        elif a.origin == types.origins["ctt"] and b.origin == types.origins["ctt"]:
            ct_add = self.cc_add_triplet(a, b)
        else:
            raise errors.DifferentTypeError(a=a.origin, b=b.origin)
        return ct_add
    
    