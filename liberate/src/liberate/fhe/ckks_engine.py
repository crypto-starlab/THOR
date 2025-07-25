import datetime
import math
import pickle
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from liberate.csprng import Csprng
from liberate.ntt import NttContext, ntt_cuda

from .context.ckks_context import CkksContext
from .create_switcher import create_switcher_cuda as csc
from .data_struct import DataStruct
from .encdec import conjugate, decode, encode, rotate
from .presets import errors, types
from .version import VERSION


class ckks_engine:
    @errors.log_error
    def __init__(self, devices: list[int] = None, verbose: bool = False,
                 bias_guard: bool = True, norm: str = 'forward', **ctx_params):
        """
            buffer_bit_length=62,
            scale_bits=40,
            logN=15,
            num_scales=None,
            num_special_primes=2,
            sigma=3.2,
            uniform_tenary_secret=True,
            cache_folder='cache/',
            security_bits=128,
            quantum='post_quantum',
            distribution='uniform',
            read_cache=True,
            save_cache=True,
            verbose=False
        """

        self.bias_guard = bias_guard

        self.norm = norm

        self.version = VERSION

        self.__ctx = CkksContext(**ctx_params)
        self.__ntt = NttContext(self.ctx, devices=devices, verbose=verbose)

        self.__num_levels = self.ntt.num_levels - 1

        self.__num_slots = self.ctx.N // 2

        rng_repeats = max(self.ntt.num_special_primes, 2)
        self.__rng = Csprng(self.ntt.ctx.N, [len(di) for di in self.ntt.p.d], rng_repeats, devices=self.ntt.devices)

        self.__int_scale = 2 ** self.ctx.scale_bits
        self.__scale = np.float64(self.int_scale)

        qstr = ','.join([str(qi) for qi in self.ctx.q])
        hashstr = (self.ctx.generation_string + "_" + qstr).encode("utf-8")
        self.hash = sha256(bytes(hashstr)).hexdigest()

        self.make_adjustments_and_corrections()

        self.device0 = self.ntt.devices[0]

        self.make_mont_PR()

        self.reserve_ksk_buffers()

        self.create_ksk_rescales()

        self.alloc_parts()

        self.leveled_devices()

        self.create_rescale_scales()

        self.__galois_deltas = [2 ** i for i in range(self.ctx.logN - 1)]

        self.dispatch_dict_mult = {
            (DataStruct, DataStruct): self.auto_cc_mult,
            (list, DataStruct): self.mc_mult,
            (np.ndarray, DataStruct): self.mc_mult,
            (DataStruct, np.ndarray): self.cm_mult,
            (DataStruct, list): self.cm_mult,
            (float, DataStruct): self.scalar_mult,
            (DataStruct, float): self.mult_scalar,
            (int, DataStruct): self.int_scalar_mult,
            (DataStruct, int): self.mult_int_scalar
        }

        self.dispatch_dict_add = {
            (DataStruct, DataStruct): self.auto_cc_add,
            (list, DataStruct): self.mc_add,
            (np.ndarray, DataStruct): self.mc_add,
            (DataStruct, np.ndarray): self.cm_add,
            (DataStruct, list): self.cm_add,
            (float, DataStruct): self.scalar_add,
            (DataStruct, float): self.add_scalar,
            (int, DataStruct): self.scalar_add,
            (DataStruct, int): self.add_scalar
        }

        self.dispatch_dict_sub = {
            (DataStruct, DataStruct): self.auto_cc_sub,
            (list, DataStruct): self.mc_sub,
            (np.ndarray, DataStruct): self.mc_sub,
            (DataStruct, np.ndarray): self.cm_sub,
            (DataStruct, list): self.cm_sub,
            (float, DataStruct): self.scalar_sub,
            (DataStruct, float): self.sub_scalar,
            (int, DataStruct): self.scalar_sub,
            (DataStruct, int): self.sub_scalar
        }
        # Initialize create_switcher plans.
        self.initialize_key_switching_plan()
        self.preallocate_create_switcher_streams()

        # This engine is not destroyed, at least yet.
        self.destroyed = False

    def strcp(self, origin):
        return "".join(list(origin).copy())

    # -------------------------------------------------------------------------------------------
    # Init create_switcher plans.
    # -------------------------------------------------------------------------------------------

    def initialize_key_switching_plan(self):
        self.plan_ids = []
        for level in range(self.num_levels):
            plan_id = self.create_switcher_plan(level)
            self.plan_ids.append(plan_id)

    def preallocate_create_switcher_streams(self):
        level = 0
        len_devices = len(self.ntt.devices)

        num_parts = []
        for src_device_id in range(len_devices):
            part_len = len(self.ntt.p.p[level][src_device_id])
            num_parts.append(part_len)

        self.streams_id = csc.preallocate_streams(num_parts, self.ntt.devices)

    # -------------------------------------------------------------------------------------------
    # Safe dealloc.
    # -------------------------------------------------------------------------------------------

    # We need to clean out the memory before exiting.

    def __del__(self):
        if not self.destroyed:
            for level in range(self.num_levels):
                csc.delete_cs_params_ptr(self.plan_ids[level])
            csc.destroy_streams(self.streams_id)

            # Destroy big attributes.
            del self.ctx
            del self.ntt
            del self.ksk_buffers

            # Release cuda cache.
            torch.cuda.empty_cache()

    # Python doesn't do a good job initiating del.
    # The reference count may not go to zero even after issuing
    # del engine.
    # Hence, we need forced method.

    def destroy(self):
        self.__del__()
        self.destroyed = True
        torch.cuda.empty_cache()

    # -------------------------------------------------------------------------------------------
    # Getter
    # -------------------------------------------------------------------------------------------
    @property
    def ctx(self):
        return self.__ctx

    @property
    def ntt(self):
        return self.__ntt

    @property
    def rng(self):
        return self.__rng

    @property
    def num_levels(self) -> int:
        return self.__num_levels

    @property
    def num_slots(self) -> int:
        return self.__num_slots

    @property
    def scale(self) -> np.float64:
        return self.__scale

    @property
    def int_scale(self) -> int:
        return self.__int_scale

    @property
    def galois_deltas(self):
        return self.__galois_deltas

    #
    # @property
    # def dispatch_dict_mult(self):
    #     return copy.deepcopy(self.__dispatch_dict_mult)
    #
    # @property
    # def dispatch_dict_add(self):
    #     return copy.deepcopy(self.__dispatch_dict_add)
    #
    # @property
    # def dispatch_dict_sub(self):
    #     return copy.deepcopy(self.__dispatch_dict_sub)
    #

    @ctx.deleter
    def ctx(self):
        del self.__ctx

    @ntt.deleter
    def ntt(self):
        del self.__ntt

    @rng.deleter
    def rng(self):
        del self.__rng

    # -------------------------------------------------------------------------------------------
    # Various pre-calculations.
    # -------------------------------------------------------------------------------------------
    def create_rescale_scales(self):
        self.rescale_scales = []
        for level in range(self.num_levels):
            self.rescale_scales.append([])

            for device_id in range(self.ntt.num_devices):
                dest_level = self.ntt.p.destination_arrays[level]

                if device_id < len(dest_level):
                    dest = dest_level[device_id]
                    rescaler_device_id = self.ntt.p.rescaler_loc[level]
                    m0 = self.ctx.q[level]

                    if rescaler_device_id == device_id:
                        m = [self.ctx.q[i] for i in dest[1:]]
                    else:
                        m = [self.ctx.q[i] for i in dest]

                    scales = [(pow(m0, -1, mi) * self.ctx.R) % mi for mi in m]

                    scales = torch.tensor(scales,
                                          dtype=self.ctx.torch_dtype,
                                          device=self.ntt.devices[device_id])
                    self.rescale_scales[level].append(scales)

    def leveled_devices(self):
        self.len_devices = []
        for level in range(self.num_levels):
            self.len_devices.append(len([[a] for a in self.ntt.p.p[level] if len(a) > 0]))

        self.neighbor_devices = []
        for level in range(self.num_levels):
            self.neighbor_devices.append([])
            len_devices_at = self.len_devices[level]
            available_devices_ids = range(len_devices_at)
            for src_device_id in available_devices_ids:
                neighbor_devices_at = [
                    device_id for device_id in available_devices_ids if device_id != src_device_id
                ]
                self.neighbor_devices[level].append(neighbor_devices_at)

    def alloc_parts(self):
        self.parts_alloc = []
        for level in range(self.num_levels):
            num_parts = [len(parts) for parts in self.ntt.p.p[level]]
            parts_alloc = [
                alloc[-num_parts[di] - 1:-1] for di, alloc in enumerate(self.ntt.p.part_allocations)
            ]
            self.parts_alloc.append(parts_alloc)

        self.stor_ids = []
        for level in range(self.num_levels):
            self.stor_ids.append([])
            alloc = self.parts_alloc[level]
            min_id = min([min(a) for a in alloc if len(a) > 0])
            for device_id in range(self.ntt.num_devices):
                global_ids = self.parts_alloc[level][device_id]
                new_ids = [i - min_id for i in global_ids]
                self.stor_ids[level].append(new_ids)

    def create_ksk_rescales(self):
        R = self.ctx.R
        P = self.ctx.q[-self.ntt.num_special_primes:][::-1]
        m = self.ctx.q
        PiR = [[(pow(Pj, -1, mi) * R) % mi for mi in m[:-P_ind - 1]] for P_ind, Pj in enumerate(P)]

        self.PiRs = []

        level = 0
        self.PiRs.append([])

        for P_ind in range(self.ntt.num_special_primes):
            self.PiRs[level].append([])

            for device_id in range(self.ntt.num_devices):
                dest = self.ntt.p.destination_arrays_with_special[level][device_id]
                PiRi = [PiR[P_ind][i] for i in dest[:-P_ind - 1]]
                PiRi = torch.tensor(PiRi,
                                    device=self.ntt.devices[device_id],
                                    dtype=self.ctx.torch_dtype)
                self.PiRs[level][P_ind].append(PiRi)

        for level in range(1, self.num_levels):
            self.PiRs.append([])

            for P_ind in range(self.ntt.num_special_primes):

                self.PiRs[level].append([])

                for device_id in range(self.ntt.num_devices):
                    start = self.ntt.starts[level][device_id]
                    PiRi = self.PiRs[0][P_ind][device_id][start:]

                    self.PiRs[level][P_ind].append(PiRi)

    def reserve_ksk_buffers(self):
        self.ksk_buffers = []
        for device_id in range(self.ntt.num_devices):
            self.ksk_buffers.append([])
            for part_id in range(len(self.ntt.p.p[0][device_id])):
                buffer = torch.empty(
                    [self.ntt.num_special_primes, self.ctx.N],
                    dtype=self.ctx.torch_dtype
                ).pin_memory()
                self.ksk_buffers[device_id].append(buffer)

    def make_mont_PR(self):
        P = math.prod(self.ntt.ctx.q[-self.ntt.num_special_primes:])
        R = self.ctx.R
        PR = P * R
        self.mont_PR = []
        for device_id in range(self.ntt.num_devices):
            dest = self.ntt.p.destination_arrays[0][device_id]
            m = [self.ctx.q[i] for i in dest]
            PRm = [PR % mi for mi in m]
            PRm = torch.tensor(PRm,
                               device=self.ntt.devices[device_id],
                               dtype=self.ctx.torch_dtype)
            self.mont_PR.append(PRm)

    def make_adjustments_and_corrections(self):

        self.alpha = [((2 ** round(math.log2(q))) / np.float64(q)) ** 2 for q in self.ctx.q[:self.ctx.num_scales]]
        self.deviations = [1]
        for al in self.alpha:
            self.deviations.append(self.deviations[-1] ** 2 * al)

        self.final_q_ind = [da[0][0] for da in self.ntt.p.destination_arrays[:-1]]
        self.final_q = [self.ctx.q[ind] for ind in self.final_q_ind]
        self.final_alpha = [(self.scale / np.float64(q)) for q in self.final_q]
        self.corrections = [1 / (d * fa) for d, fa in zip(self.deviations, self.final_alpha)]

        self.base_prime = self.ctx.q[self.ntt.p.base_prime_idx]

        self.final_scalar = []
        for qi, q in zip(self.final_q_ind, self.final_q):
            scalar = (pow(q, -1, self.base_prime) * self.ctx.R) % self.base_prime
            scalar = torch.tensor([scalar],
                                  device=self.ntt.devices[0],
                                  dtype=self.ctx.torch_dtype)
            self.final_scalar.append(scalar)

        r = self.ctx.R

        self.new_final_scalar = [
            torch.tensor(
                [
                    (pow(self.ctx.q[final_q_ind], -1, base_prime) * r) % base_prime
                    for base_prime in self.ctx.q[final_q_ind + 1: -self.ctx.num_special_primes]
                ],
                device=self.ntt.devices[0],
                dtype=self.ctx.torch_dtype
            ) for final_q_ind in range(self.ntt.num_ordinary_primes - 1)
        ]

    # -------------------------------------------------------------------------------------------
    # Create switcher plans.
    # -------------------------------------------------------------------------------------------

    def pre_extend_plan(self, device_id, level, part_id):
        # param_parts contain only the ordinary parts.
        # Hence, loop around it.
        # text_parts contain special primes.
        text_part = self.ntt.p.parts[level][device_id][part_id]
        param_part = self.ntt.p.p[level][device_id][part_id]

        # What is my partition length?
        alpha = len(text_part)

        # part_info contains the basic information about this partition,
        # including the storage_id.
        part_info = [alpha, text_part[0]]

        # We need separate lists for mont_enter, _2q, Y_scalar, state_mont_pack,
        # state_2q, and L_scalar.
        mont_packs = []
        Y_scalars = []
        state_mont_packs = []
        L_scalars = []

        key = tuple(param_part)

        for i in range(alpha - 1):
            mont_pack = self.ntt.parts_pack[device_id][param_part[i + 1],]['mont_pack']
            Y_scalar = self.ntt.parts_pack[device_id][key]['Y_scalar'][i][None]

            # mont_pack and _2q are lists of list to account for multiple devices.
            # However, here the packs are for a single device.
            # We don't want to complicate the API too much, and hence unpack the
            # last list dimension.
            mont_pack = [pack[0] for pack in mont_pack]
            mont_packs.append(mont_pack)
            Y_scalars.append(Y_scalar)

            if i + 2 < alpha:
                state_key = tuple(param_part[i + 2:])
                state_mont_pack = self.ntt.parts_pack[device_id][state_key]['mont_pack']
                L_scalar = self.ntt.parts_pack[device_id][key]['L_scalar'][i]

                # Again, unpack the state_mont_pack and the state_2q.
                state_mont_pack = [pack[0] for pack in state_mont_pack]

                state_mont_packs.append(state_mont_pack)
                L_scalars.append(L_scalar)

        return part_info, mont_packs, Y_scalars, state_mont_packs, L_scalars

    def extend_plan(self, device_id, level, part_id):

        # Basic info.
        # Device lens and neighbor devices.
        # len_devices = self.len_devices[level]
        neighbor_device_ids = self.neighbor_devices[level][device_id]
        device_ids = [device_id] + neighbor_device_ids

        # Iterate over source device ids, and then part ids.
        # num_parts = sum([len(alloc) for alloc in ksk_alloc])

        # CPU states is the pinned memory buffers.
        part = self.ntt.p.p[level][device_id]
        alpha = len(part)
        CPU_state = self.ksk_buffers[device_id][part_id][:alpha]

        # Extend basis.
        # At this point the state has been transferred over, or
        # resides in the emanating device.
        L_enters = []
        for did in device_ids:

            # Generate the search key to find the L_enter.
            part = self.ntt.p.p[level][device_id][part_id]
            key = tuple(part)

            # Get L_enter.
            # device_id is the source and did is the target.
            L_enter = self.ntt.parts_pack[device_id][key]['L_enter'][did]

            # L_enter can be None.
            if L_enter is not None:
                # L_enter covers the whole rns range.
                # Start from the leveled start.
                L_start = self.ntt.starts[level][did]

                # Extract the region of interest.
                L_enter = [data[L_start:] for data in L_enter]
                L_enters.append(L_enter)
            else:
                L_enters.append([])

        ksk_loc = self.parts_alloc[level][device_id][part_id]

        return CPU_state, L_enters, ksk_loc

    def later_part_plan(self, device_id, level):
        # Basic info.
        # Device lens and neighbor devices.
        # len_devices = self.len_devices[level]

        # We will branch out to all the available GPUs in the devices list.
        # Again, we want the actual device ids in CUDA parlance.
        neighbor_device_ids = self.neighbor_devices[level][device_id]
        neighbor_devices = [self.ntt.devices[did] for did in neighbor_device_ids]

        # run_devices include my device.
        my_device = self.ntt.devices[device_id]
        run_devices = [my_device] + neighbor_devices
        device_ids = [device_id] + neighbor_device_ids

        # Extend basis.
        # At this point the state has been transferred over, or
        # resides in the emanating device.
        did = device_id

        # The default indexing conversion for extracting packs
        # is [mult_type][level][0], for multiple devices at once.
        # When it is accessed as [device_id][level][part],
        # It gives the pack for the specific device.
        # part = -1 for ordinary and -2 for special if negative.
        # Otherwise, it denotes the part_id.

        # For mont_enter.
        # Don't forget to remove the redundant indirection.
        # We are dealing with one device.
        Rs_prepack = self.ntt.Rs_prepack[did][level][-2][0]
        mont_prepack = self.ntt.mont_prepack[did][level][-2]
        mont_prepack = [data[0] for data in mont_prepack]

        # We need _2q for mont_add.
        _2q = self.ntt._2q_prepack[did][level][-2][0]

        # Fuse later part.
        ntt_prepack = self.ntt.ntt_prepack[did][level][-2]
        ntt_prepack = [data[0] for data in ntt_prepack]

        intt_prepack = self.ntt.intt_prepack[did][level][-2]
        intt_prepack = [data[0] for data in intt_prepack]

        return run_devices, device_ids, Rs_prepack, mont_prepack, _2q, ntt_prepack, intt_prepack

    def create_switcher_plan(self, level):

        # Device lens and neighbor devices.
        len_devices = self.len_devices[level]
        # neighbor_devices = self.neighbor_devices[level]

        # Iterate over source device ids, and then part ids.
        # num_parts = sum([len(alloc) for alloc in ksk_alloc])

        # 1. Generate pre_extension plans.
        #    We will hand over the list of plans as is.
        #    That means the execution order of pre_extend will have to
        #    follow exactly the order of plan generation.
        part_plan = [[] for _ in range(len_devices)]
        mont_plan = [[] for _ in range(len_devices)]
        Y_plan = [[] for _ in range(len_devices)]
        state_mont_plan = [[] for _ in range(len_devices)]
        L_scalar_plan = [[] for _ in range(len_devices)]

        run_device_plan = [[] for _ in range(len_devices)]
        device_id_plan = [[] for _ in range(len_devices)]
        CPU_state_plan = [[] for _ in range(len_devices)]
        Rs_prepack_plan = [[] for _ in range(len_devices)]
        mont_prepack_plan = [[] for _ in range(len_devices)]
        L_enter_plan = [[] for _ in range(len_devices)]
        _2q_plan = [[] for _ in range(len_devices)]
        ntt_prepack_plan = [[] for _ in range(len_devices)]
        intt_prepack_plan = [[] for _ in range(len_devices)]
        ksk_loc_plan = [[] for _ in range(len_devices)]

        for src_device_id in range(len_devices):
            for part_id in range(len(self.ntt.p.p[level][src_device_id])):
                part_info, mont_packs, Y_scalars, state_mont_packs, L_scalars = \
                    self.pre_extend_plan(src_device_id, level, part_id)

                part_plan[src_device_id].append(part_info)
                mont_plan[src_device_id].append(mont_packs)
                Y_plan[src_device_id].append(Y_scalars)
                state_mont_plan[src_device_id].append(state_mont_packs)
                L_scalar_plan[src_device_id].append(L_scalars)

                CPU_state, L_enters, ksk_loc = self.extend_plan(src_device_id, level, part_id)

                CPU_state_plan[src_device_id].append(CPU_state)
                L_enter_plan[src_device_id].append(L_enters)
                ksk_loc_plan[src_device_id].append(ksk_loc)

            run_devices, device_ids, Rs_prepack, mont_prepack, _2q, \
                ntt_prepack, intt_prepack = self.later_part_plan(src_device_id, level)

            run_device_plan[src_device_id] = run_devices
            device_id_plan[src_device_id] = device_ids
            Rs_prepack_plan[src_device_id] = Rs_prepack
            mont_prepack_plan[src_device_id] = mont_prepack
            _2q_plan[src_device_id] = _2q
            ntt_prepack_plan[src_device_id] = ntt_prepack
            intt_prepack_plan[src_device_id] = intt_prepack

        # We also need to know where the data tensor channel starts
        # at this level.
        # Remember, the devices must be the actual CUDA devices.
        ksk_starts_plan = self.ntt.starts[level]

        # For the final division and constriction of the RNS basis,
        # We need P_inv's. The values are stored in P_inv * R format
        # in Pir.
        PiRi_plan = []
        for P_ind in range(self.ntt.num_special_primes):
            PiRi_plan.append(self.PiRs[level][P_ind])

        # We need the state buffers.
        # Allocating memory in CUDA is expensive, pre-allocating from the pytorch memory pool is not.
        states = [
            [torch.zeros([len(da), self.ctx.N], dtype=self.ctx.torch_dtype, device=device)
             for device in self.ntt.devices]
            for da in self.ntt.p.destination_arrays[0]]

        # extended as well.
        extended_part = lambda level: [
            torch.empty(
                [len(da), self.ctx.N], device=device, dtype=self.ctx.torch_dtype)
            for da, device in zip(self.ntt.p.destination_arrays_with_special[level], self.ntt.devices)
        ]
        extended = [extended_part(0) for _ in range(self.ntt.p.num_partitions + 1)]

        plan = [
            part_plan,
            mont_plan,
            Y_plan,
            state_mont_plan,
            L_scalar_plan,
            run_device_plan,
            device_id_plan,
            CPU_state_plan,
            Rs_prepack_plan,
            mont_prepack_plan,
            L_enter_plan,
            _2q_plan,
            ntt_prepack_plan,
            intt_prepack_plan,
            ksk_loc_plan,
            ksk_starts_plan,
            PiRi_plan,
            states,
            extended
        ]

        plan_id = csc.create_cs_params_ptr(*plan)

        return plan_id

    # -------------------------------------------------------------------------------------------
    # Create switcher plans.
    # -------------------------------------------------------------------------------------------

    def pre_extend_plan(self, device_id, level, part_id):
        # param_parts contain only the ordinary parts.
        # Hence, loop around it.
        # text_parts contain special primes.
        text_part = self.ntt.p.parts[level][device_id][part_id]
        param_part = self.ntt.p.p[level][device_id][part_id]

        # What is my partition length?
        alpha = len(text_part)

        # part_info contains the basic information about this partition,
        # including the storage_id.
        part_info = [alpha, text_part[0]]

        # We need separate lists for mont_enter, _2q, Y_scalar, state_mont_pack,
        # state_2q, and L_scalar.
        mont_packs = []
        Y_scalars = []
        state_mont_packs = []
        L_scalars = []

        key = tuple(param_part)

        for i in range(alpha - 1):
            mont_pack = self.ntt.parts_pack[device_id][param_part[i + 1],]['mont_pack']
            Y_scalar = self.ntt.parts_pack[device_id][key]['Y_scalar'][i][None]

            # mont_pack and _2q are lists of list to account for multiple devices.
            # However, here the packs are for a single device.
            # We don't want to complicate the API too much, and hence unpack the
            # last list dimension.
            mont_pack = [pack[0] for pack in mont_pack]
            mont_packs.append(mont_pack)
            Y_scalars.append(Y_scalar)

            if i + 2 < alpha:
                state_key = tuple(param_part[i + 2:])
                state_mont_pack = self.ntt.parts_pack[device_id][state_key]['mont_pack']
                L_scalar = self.ntt.parts_pack[device_id][key]['L_scalar'][i]

                # Again, unpack the state_mont_pack and the state_2q.
                state_mont_pack = [pack[0] for pack in state_mont_pack]

                state_mont_packs.append(state_mont_pack)
                L_scalars.append(L_scalar)

        return part_info, mont_packs, Y_scalars, state_mont_packs, L_scalars

    def extend_plan(self, device_id, level, part_id):

        # Basic info.
        # Device lens and neighbor devices.
        # len_devices = self.len_devices[level]
        neighbor_device_ids = self.neighbor_devices[level][device_id]
        device_ids = [device_id] + neighbor_device_ids

        # Iterate over source device ids, and then part ids.
        # num_parts = sum([len(alloc) for alloc in ksk_alloc])

        # CPU states is the pinned memory buffers.
        part = self.ntt.p.p[level][device_id]
        alpha = len(part)
        CPU_state = self.ksk_buffers[device_id][part_id][:alpha]

        # Extend basis.
        # At this point the state has been transferred over, or
        # resides in the emanating device.
        L_enters = []
        for did in device_ids:

            # Generate the search key to find the L_enter.
            part = self.ntt.p.p[level][device_id][part_id]
            key = tuple(part)

            # Get L_enter.
            # device_id is the source and did is the target.
            L_enter = self.ntt.parts_pack[device_id][key]['L_enter'][did]

            # L_enter can be None.
            if L_enter is not None:
                # L_enter covers the whole rns range.
                # Start from the leveled start.
                L_start = self.ntt.starts[level][did]

                # Extract the region of interest.
                L_enter = [data[L_start:] for data in L_enter]
                L_enters.append(L_enter)
            else:
                L_enters.append([])

        ksk_loc = self.parts_alloc[level][device_id][part_id]

        return CPU_state, L_enters, ksk_loc

    def later_part_plan(self, device_id, level):
        # Basic info.
        # Device lens and neighbor devices.
        # len_devices = self.len_devices[level]

        # We will branch out to all the available GPUs in the devices list.
        # Again, we want the actual device ids in CUDA parlance.
        neighbor_device_ids = self.neighbor_devices[level][device_id]
        neighbor_devices = [self.ntt.devices[did] for did in neighbor_device_ids]

        # run_devices include my device.
        my_device = self.ntt.devices[device_id]
        run_devices = [my_device] + neighbor_devices
        device_ids = [device_id] + neighbor_device_ids

        # Extend basis.
        # At this point the state has been transferred over, or
        # resides in the emanating device.
        did = device_id

        # The default indexing conversion for extracting packs
        # is [mult_type][level][0], for multiple devices at once.
        # When it is accessed as [device_id][level][part],
        # It gives the pack for the specific device.
        # part = -1 for ordinary and -2 for special if negative.
        # Otherwise, it denotes the part_id.

        # For mont_enter.
        # Don't forget to remove the redundant indirection.
        # We are dealing with one device.
        Rs_prepack = self.ntt.Rs_prepack[did][level][-2][0]
        mont_prepack = self.ntt.mont_prepack[did][level][-2]
        mont_prepack = [data[0] for data in mont_prepack]

        # We need _2q for mont_add.
        _2q = self.ntt._2q_prepack[did][level][-2][0]

        # Fuse later part.
        ntt_prepack = self.ntt.ntt_prepack[did][level][-2]
        ntt_prepack = [data[0] for data in ntt_prepack]

        intt_prepack = self.ntt.intt_prepack[did][level][-2]
        intt_prepack = [data[0] for data in intt_prepack]

        return run_devices, device_ids, Rs_prepack, mont_prepack, _2q, ntt_prepack, intt_prepack

    def create_switcher_plan(self, level):

        # Device lens and neighbor devices.
        len_devices = self.len_devices[level]
        # neighbor_devices = self.neighbor_devices[level]

        # Iterate over source device ids, and then part ids.
        # num_parts = sum([len(alloc) for alloc in ksk_alloc])

        # 1. Generate pre_extension plans.
        #    We will hand over the list of plans as is.
        #    That means the execution order of pre_extend will have to
        #    follow exactly the order of plan generation.
        part_plan = [[] for _ in range(len_devices)]
        mont_plan = [[] for _ in range(len_devices)]
        Y_plan = [[] for _ in range(len_devices)]
        state_mont_plan = [[] for _ in range(len_devices)]
        L_scalar_plan = [[] for _ in range(len_devices)]

        run_device_plan = [[] for _ in range(len_devices)]
        device_id_plan = [[] for _ in range(len_devices)]
        CPU_state_plan = [[] for _ in range(len_devices)]
        Rs_prepack_plan = [[] for _ in range(len_devices)]
        mont_prepack_plan = [[] for _ in range(len_devices)]
        L_enter_plan = [[] for _ in range(len_devices)]
        _2q_plan = [[] for _ in range(len_devices)]
        ntt_prepack_plan = [[] for _ in range(len_devices)]
        intt_prepack_plan = [[] for _ in range(len_devices)]
        ksk_loc_plan = [[] for _ in range(len_devices)]

        for src_device_id in range(len_devices):
            for part_id in range(len(self.ntt.p.p[level][src_device_id])):
                part_info, mont_packs, Y_scalars, state_mont_packs, L_scalars = \
                    self.pre_extend_plan(src_device_id, level, part_id)

                part_plan[src_device_id].append(part_info)
                mont_plan[src_device_id].append(mont_packs)
                Y_plan[src_device_id].append(Y_scalars)
                state_mont_plan[src_device_id].append(state_mont_packs)
                L_scalar_plan[src_device_id].append(L_scalars)

                CPU_state, L_enters, ksk_loc = self.extend_plan(src_device_id, level, part_id)

                CPU_state_plan[src_device_id].append(CPU_state)
                L_enter_plan[src_device_id].append(L_enters)
                ksk_loc_plan[src_device_id].append(ksk_loc)

            run_devices, device_ids, Rs_prepack, mont_prepack, _2q, \
                ntt_prepack, intt_prepack = self.later_part_plan(src_device_id, level)

            run_device_plan[src_device_id] = run_devices
            device_id_plan[src_device_id] = device_ids
            Rs_prepack_plan[src_device_id] = Rs_prepack
            mont_prepack_plan[src_device_id] = mont_prepack
            _2q_plan[src_device_id] = _2q
            ntt_prepack_plan[src_device_id] = ntt_prepack
            intt_prepack_plan[src_device_id] = intt_prepack

        # We also need to know where the data tensor channel starts
        # at this level.
        # Remember, the devices must be the actual CUDA devices.
        ksk_starts_plan = self.ntt.starts[level]

        # For the final division and constriction of the RNS basis,
        # We need P_inv's. The values are stored in P_inv * R format
        # in Pir.
        PiRi_plan = []
        for P_ind in range(self.ntt.num_special_primes):
            PiRi_plan.append(self.PiRs[level][P_ind])

        # We need the state buffers.
        # Allocating memory in CUDA is expensive, pre-allocating from the pytorch memory pool is not.
        states = [
            [torch.zeros([len(da), self.ctx.N], dtype=self.ctx.torch_dtype, device=device)
             for device in self.ntt.devices]
            for da in self.ntt.p.destination_arrays[0]]

        # extended as well.
        extended_part = lambda level: [
            torch.empty(
                [len(da), self.ctx.N], device=device, dtype=self.ctx.torch_dtype)
            for da, device in zip(self.ntt.p.destination_arrays_with_special[level], self.ntt.devices)
        ]
        extended = [extended_part(0) for _ in range(self.ntt.p.num_partitions + 1)]

        plan = [
            part_plan,
            mont_plan,
            Y_plan,
            state_mont_plan,
            L_scalar_plan,
            run_device_plan,
            device_id_plan,
            CPU_state_plan,
            Rs_prepack_plan,
            mont_prepack_plan,
            L_enter_plan,
            _2q_plan,
            ntt_prepack_plan,
            intt_prepack_plan,
            ksk_loc_plan,
            ksk_starts_plan,
            PiRi_plan,
            states,
            extended
        ]

        plan_id = csc.create_cs_params_ptr(*plan)

        return plan_id

    # -------------------------------------------------------------------------------------------
    # Example generation.
    # -------------------------------------------------------------------------------------------

    def absmax_error(self, x, y):
        if isinstance(x[0], np.complex128) and isinstance(y[0], np.complex128):
            r = np.abs(x.real - y.real).max() + np.abs(x.imag - y.imag).max() * 1j
        else:
            r = np.abs(np.array(x) - np.array(y)).max()
        return r

    def integral_bits_available(self):
        base_prime = self.base_prime
        max_bits = math.floor(math.log2(base_prime))
        integral_bits = max_bits - self.ctx.scale_bits
        return integral_bits

    @errors.log_error
    def example(self, amin=None, amax=None, decimal_places: int = 10) -> np.array:
        if amin is None:
            amin = -(2 ** self.integral_bits_available())

        if amax is None:
            amax = 2 ** self.integral_bits_available()

        base = 10 ** decimal_places
        a = np.random.randint(amin * base, amax * base, self.ctx.N // 2) / base
        b = np.random.randint(amin * base, amax * base, self.ctx.N // 2) / base

        sample = a + b * 1j

        return sample

    # -------------------------------------------------------------------------------------------
    # Encode/Decode
    # -------------------------------------------------------------------------------------------

    def padding(self, m):
        try:
            m_len = len(m)
            padding_result = np.pad(m, (0, self.num_slots - m_len), constant_values=(0, 0))
        except TypeError as e:
            m_len = len([m])
            padding_result = np.pad([m], (0, self.num_slots - m_len), constant_values=(0, 0))
        except Exception as e:
            raise Exception("[Error] encoding Padding Error.")
        return padding_result

    @errors.log_error
    def encode(self, m, level: int = 15, padding=True) -> list[torch.Tensor]:
        """
            Encode a plain message m, using an encoding function.
            Note that the encoded plain text is pre-permuted to yield cyclic rotation.
        """
        
        deviation = self.deviations[level]
        if padding:
            m = self.padding(m)
        encoded = [encode(m, scale=self.scale, rng=self.rng,
                          device=self.device0,
                          deviation=deviation, norm=self.norm)]

        pt_buffer = self.ksk_buffers[0][0][0]
        pt_buffer.copy_(encoded[-1])
        for dev_id in range(1, self.ntt.num_devices):
            encoded.append(pt_buffer.cuda(self.ntt.devices[dev_id]))
        return encoded

    @errors.log_error
    def encode_bootstrapping(self, m, level: int = 0, padding=True) -> list[torch.Tensor]:
        """
            Encode a plain message m, using an encoding function.
            Note that the encoded plain text is pre-permuted to yield cyclic rotation.
        """
        deviation = np.sqrt(self.deviations[level + 1])
        if padding:
            m = self.padding(m)
        encoded = [encode(m, scale=2 ** 59, rng=self.rng,
                          device=self.device0,
                          deviation=deviation, norm=self.norm)]

        pt_buffer = self.ksk_buffers[0][0][0]
        pt_buffer.copy_(encoded[-1])
        for dev_id in range(1, self.ntt.num_devices):
            encoded.append(pt_buffer.cuda(self.ntt.devices[dev_id]))
        return encoded

    @errors.log_error
    def decode(self, m: list[torch.Tensor] | np.ndarray, level:int=15, is_real: bool = False) -> np.ndarray:
        """
            Base prime is located at -1 of the RNS channels in GPU0.
            Assuming this is an orginary RNS deinclude_special.
        """
        correction = self.corrections[level]
        decoded = decode(m[0].squeeze(), scale=self.scale, correction=correction, norm=self.norm)
        m = decoded[:self.ctx.N // 2].cpu().numpy()
        if is_real:
            m = m.real
        return m

    # -------------------------------------------------------------------------------------------
    # secret key/public key generation.
    # -------------------------------------------------------------------------------------------

    @errors.log_error
    def create_secret_key(self, include_special: bool = True) -> DataStruct:
        uniform_ternary = self.rng.randint(amax=3, shift=-1, repeats=1)

        mult_type = -2 if include_special else -1
        unsigned_ternary = self.ntt.tile_unsigned(uniform_ternary, lvl=0, mult_type=mult_type)
        self.ntt.enter_ntt(unsigned_ternary, 0, mult_type)

        return DataStruct(
            data=unsigned_ternary,
            include_special=include_special,
            montgomery_state=True,
            ntt_state=True,
            origin=types.origins["sk"],
            level_calc=0,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    @errors.log_error
    def create_public_key(self, sk: DataStruct, include_special: bool = False,
                          crs: list[torch.Tensor] = None) -> DataStruct:
        """
            Generates a public key against the secret key sk.
            pk = -a * sk + e = e - a * sk
        """
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if include_special and not sk.include_special:
            raise errors.SecretKeyNotIncludeSpecialPrime()

        # Set the mult_type
        mult_type = -2 if include_special else -1

        # Generate errors for the ordinary case.
        level = 0
        e = self.rng.discrete_gaussian(repeats=1)
        e = self.ntt.tile_unsigned(e, level, mult_type)

        self.ntt.enter_ntt(e, level, mult_type)
        repeats = self.ctx.num_special_primes if sk.include_special else 0

        # Applying mont_mult in the order of 'a', sk will
        if crs is None:
            crs = self.rng.randint(
                self.ntt.q_prepack[mult_type][level][0],
                repeats=repeats
            )

        sa = self.ntt.mont_mult(crs, sk.data, 0, mult_type)
        pk0 = self.ntt.mont_sub(e, sa, 0, mult_type)

        return DataStruct(
            data=(pk0, crs),
            include_special=include_special,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["pk"],
            level_calc=0,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    # -------------------------------------------------------------------------------------------
    # Encrypt/Decrypt
    # -------------------------------------------------------------------------------------------

    @errors.log_error
    def encrypt(self, pt: list[torch.Tensor], pk: DataStruct, level: int = 15) -> DataStruct:
        """
            We again, multiply pt by the scale.
            Since pt is already multiplied by the scale,
            the multiplied pt no longer can be stored
            in a single RNS channel.
            That means we are forced to do the multiplication
            in full RNS domain.
            Note that we allow encryption at
            levels other than 0, and that will take care of multiplying
            the deviation factors.
        @param pt:
        @param pk:
        @param level:
        @return:
        """
        if pk.origin != types.origins["pk"]:
            raise errors.NotMatchType(origin=pk.origin, to=types.origins["pk"])

        mult_type = -2 if pk.include_special else -1

        e0e1 = self.rng.discrete_gaussian(repeats=2)

        e0 = [e[0] for e in e0e1]
        e1 = [e[1] for e in e0e1]

        e0_tiled = self.ntt.tile_unsigned(e0, level, mult_type)
        e1_tiled = self.ntt.tile_unsigned(e1, level, mult_type)

        pt_tiled = self.ntt.tile_unsigned(pt, level, mult_type)
        self.ntt.mont_enter_scale(pt_tiled, level, mult_type)
        self.ntt.mont_redc(pt_tiled, level, mult_type)
        pte0 = self.ntt.mont_add(pt_tiled, e0_tiled, level, mult_type)

        start = self.ntt.starts[level]
        pk0 = [pk.data[0][di][start[di]:] for di in range(self.ntt.num_devices)]
        pk1 = [pk.data[1][di][start[di]:] for di in range(self.ntt.num_devices)]

        v = self.rng.randint(amax=2, shift=0, repeats=1)

        v = self.ntt.tile_unsigned(v, level, mult_type)
        self.ntt.enter_ntt(v, level, mult_type)

        vpk0 = self.ntt.mont_mult(v, pk0, level, mult_type)
        vpk1 = self.ntt.mont_mult(v, pk1, level, mult_type)

        self.ntt.intt_exit(vpk0, level, mult_type)
        self.ntt.intt_exit(vpk1, level, mult_type)

        ct0 = self.ntt.mont_add(vpk0, pte0, level, mult_type)
        ct1 = self.ntt.mont_add(vpk1, e1_tiled, level, mult_type)

        self.ntt.reduce_2q(ct0, level, mult_type)
        self.ntt.reduce_2q(ct1, level, mult_type)

        ct = DataStruct(
            data=(ct0, ct1),
            include_special=mult_type == -2,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

        return ct

    def decrypt_triplet(self, ct_mult: DataStruct, sk: DataStruct, final_round=True) -> list[torch.Tensor]:
        if ct_mult.origin != types.origins["ctt"]:
            raise errors.NotMatchType(origin=ct_mult.origin, to=types.origins["ctt"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        if not ct_mult.ntt_state or not ct_mult.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct_mult.origin)
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        level = ct_mult.level_calc
        d0 = [ct_mult.data[0][0].clone()]
        d1 = [ct_mult.data[1][0]]
        d2 = [ct_mult.data[2][0]]

        self.ntt.intt_exit_reduce(d0, level)

        sk_data = [sk.data[0][self.ntt.starts[level][0]:]]

        d1_s = self.ntt.mont_mult(d1, sk_data, level)

        s2 = self.ntt.mont_mult(sk_data, sk_data, level)
        d2_s2 = self.ntt.mont_mult(d2, s2, level)

        self.ntt.intt_exit(d1_s, level)
        self.ntt.intt_exit(d2_s2, level)

        pt = self.ntt.mont_add(d0, d1_s, level)
        pt = self.ntt.mont_add(pt, d2_s2, level)
        self.ntt.reduce_2q(pt, level)

        base_at = -self.ctx.num_special_primes - 1 if ct_mult.include_special else -1

        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)

        # Round?
        if final_round:
            # The scaler and the base channels are guaranteed to be in the
            # device 0.
            rounding_prime = self.ntt.qlists[0][-self.ctx.num_special_primes - 2]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        return scaled

    def decrypt_double(self, ct: DataStruct, sk: DataStruct, final_round:bool=True) -> list[torch.Tensor]:
        """

        @param ct:
        @param sk:
        @param final_round:
        @return:
        """
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)
        if not sk.ntt_state or not sk.montgomery_state:
            raise errors.NotMatchDataStructState(origin=sk.origin)

        level = ct.level_calc

        ct0 = ct.data[0][0]
        sk_data = sk.data[0][self.ntt.starts[level][0]:]
        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)
        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        pt = self.ntt.mont_add([ct0], sa, level)
        self.ntt.reduce_2q(pt, level)

        if level == self.num_levels - 1:
            base_at = -self.ctx.num_special_primes - 1 if ct.include_special else -1

            base = pt[0][base_at][None, :]
            scaler = pt[0][0][None, :]

            final_scalar = self.final_scalar[level]
            scaled = self.ntt.mont_sub([base], [scaler], -1)
            self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
            self.ntt.reduce_2q(scaled, -1)
            self.ntt.make_signed(scaled, -1)

        else:  # level > self.num_levels - 1
            base = pt[0][1:]
            scaler = pt[0][0][None, :]

            final_scalar = self.new_final_scalar[level]

            scaled = base - scaler
            self.ntt.make_unsigned([scaled], lvl=level + 1)
            self.ntt.mont_enter_scalar([scaled], [final_scalar], lvl=level + 1)
            self.ntt.reduce_2q([scaled], lvl=level + 1)

            scaled = scaled[-2:]  # Two channels for decryption

            q1 = self.ctx.q[self.ntt.num_ordinary_primes - 2]
            q0 = self.ctx.q[self.ntt.num_ordinary_primes - 1]

            q1_inv_mod_q0_mont = self.new_final_scalar[self.ntt.num_ordinary_primes - 2]

            quotient = scaled[1:] - scaled[0]
            self.ntt.mont_enter_scalar([quotient], [q1_inv_mod_q0_mont], self.num_levels)
            self.ntt.reduce_2q([quotient], self.num_levels)

            M_half_div_q1 = (q0 - 1) // 2
            M_half_mod_q1 = (q1 - 1) // 2
            is_negative = torch.logical_or(
                quotient[0] > M_half_div_q1,
                torch.logical_and(
                    quotient[0] >= M_half_div_q1, scaled[0] > M_half_mod_q1
                )
            )
            is_negative = is_negative * 1

            signed_large_part = quotient[0] - is_negative * q0
            scaled = [scaled[0].type(torch.float64) + float(q1) * signed_large_part.type(torch.float64)]

        if final_round:
            rounding_prime = self.ntt.qlists[0][-self.ctx.num_special_primes - 2]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        return scaled

    def decrypt(self, ct: DataStruct, sk: DataStruct, final_round=True) -> list[torch.Tensor]:
        """
            Decrypt the cipher text ct using the secret key sk.
            Note that the final rescaling must precede the actual decryption process.
        """

        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        if ct.origin == types.origins["ctt"]:
            pt = self.decrypt_triplet(ct_mult=ct, sk=sk)
        elif ct.origin == types.origins["ct"]:
            pt = self.decrypt_double(ct=ct, sk=sk, final_round=final_round)
        else:
            raise errors.NotMatchType(origin=ct.origin, to=f"{types.origins['ct']} or {types.origins['ctt']}")

        return pt

    # -------------------------------------------------------------------------------------------
    # Key switching.
    # -------------------------------------------------------------------------------------------

    def create_key_switching_key(self, sk_from: DataStruct, sk_to: DataStruct, level=None, crs=None) -> DataStruct:
        """
            Creates a key to switch the key for sk_src to sk_dst.
        """

        if sk_from.origin != types.origins["sk"] or sk_from.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin="not a secret key", to=types.origins["sk"])
        if (not sk_from.ntt_state) or (not sk_from.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_from.origin)
        if (not sk_to.ntt_state) or (not sk_to.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_to.origin)

        if level is None:
            level = 0

        stops = self.ntt.stops[-1]
        Psk_src = [sk_from.data[di][:stops[di]].clone() for di in range(self.ntt.num_devices)]

        self.ntt.mont_enter_scalar(Psk_src, self.mont_PR, level)

        ksk = [[] for _ in range(self.ntt.p.num_partitions + 1)]

        for device_id in range(self.ntt.num_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][device_id]):
                global_part_id = self.ntt.p.part_allocations[device_id][part_id]

                a = crs[global_part_id] if crs else None
                pk = self.create_public_key(sk_to, include_special=True, crs=a)

                key = tuple(part)
                astart = part[0]
                astop = part[-1] + 1
                shard = Psk_src[device_id][astart:astop]
                pk_data = pk.data[0][device_id][astart:astop]

                _2q = self.ntt.parts_pack[device_id][key]['_2q']
                update_part = ntt_cuda.mont_add([pk_data], [shard], _2q)[0]
                pk_data.copy_(update_part, non_blocking=True)

                pk_name = f'key switch key part index {global_part_id}'
                pk = pk._replace(origin=pk_name)

                ksk[global_part_id] = pk

        return DataStruct(
            data=ksk,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ksk"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    def pre_extend(self, a, device_id, level, part_id, exit_ntt=False):
        # param_parts contain only the ordinary parts.
        # Hence, loop around it.
        # text_parts contain special primes.
        text_part = self.ntt.p.parts[level][device_id][part_id]
        param_part = self.ntt.p.p[level][device_id][part_id]

        # Carve out the partition.
        alpha = len(text_part)
        a_part = a[device_id][text_part[0]:text_part[-1] + 1]

        # Release ntt.
        if exit_ntt:
            self.ntt.intt_exit_reduce([a_part], level, device_id, part_id)

        # Prepare a state.
        # Initially, it is x[0] % m[i].
        # However, m[i] is a monotonically increasing
        # sequence, i.e., repeating the row would suffice
        # to do the job.

        # 2023-10-16, Juwhan Kim, In fact, m[i] is NOT monotonically increasing.

        state = a_part[0].repeat(alpha, 1)

        key = tuple(param_part)
        for i in range(alpha - 1):
            mont_pack = self.ntt.parts_pack[device_id][param_part[i + 1],]['mont_pack']
            _2q = self.ntt.parts_pack[device_id][param_part[i + 1],]['_2q']
            Y_scalar = self.ntt.parts_pack[device_id][key]['Y_scalar'][i][None]

            Y = (a_part[i + 1] - state[i + 1])[None, :]

            # mont_enter will take care of signedness.
            # ntt_cuda.make_unsigned([Y], _2q)
            ntt_cuda.mont_enter([Y], [Y_scalar], *mont_pack)
            # ntt_cuda.reduce_2q([Y], _2q)

            state[i + 1] = Y

            if i + 2 < alpha:
                state_key = tuple(param_part[i + 2:])
                state_mont_pack = self.ntt.parts_pack[device_id][state_key]['mont_pack']
                L_scalar = self.ntt.parts_pack[device_id][key]['L_scalar'][i]
                new_state_len = alpha - (i + 2)
                new_state = Y.repeat(new_state_len, 1)
                ntt_cuda.mont_enter([new_state], [L_scalar], *state_mont_pack)
                state[i + 2:] += new_state

        # Returned state is in plain integer format.
        return state

    def extend(self, state, device_id, level, part_id, target_device_id=None):
        # Note that device_id, level, and part_id is from
        # where the state has been originally calculated at.
        # The state can reside in a different GPU than
        # the original one.

        if target_device_id is None:
            target_device_id = device_id

        rns_len = len(
            self.ntt.p.destination_arrays_with_special[level][target_device_id])
        alpha = len(state)

        # Initialize the output
        extended = state[0].repeat(rns_len, 1)
        self.ntt.mont_enter([extended], level, target_device_id, -2)

        # Generate the search key to find the L_enter.
        part = self.ntt.p.p[level][device_id][part_id]
        key = tuple(part)

        # Extract the L_enter in the target device.
        L_enter = self.ntt.parts_pack[device_id][key]['L_enter'][target_device_id]

        # L_enter covers the whole rns range.
        # Start from the leveled start.
        start = self.ntt.starts[level][target_device_id]

        # Loop to generate.
        for i in range(alpha - 1):
            Y = state[i + 1].repeat(rns_len, 1)

            self.ntt.mont_enter_scalar([Y], [L_enter[i][start:]], level, target_device_id, -2)
            extended = self.ntt.mont_add([extended], [Y], level, target_device_id, -2)[0]

        # Returned extended is in the Montgomery format.
        return extended

    def create_switcher_fast(self, a, ksk, level, exit_ntt=False):
        if exit_ntt:
            self.ntt.intt_exit_reduce(a, level, -1)

        # The output.
        num_chans = [len(da) for da in self.ntt.p.destination_arrays_with_special[level]]
        switcher = [[torch.zeros([C, self.ctx.N], dtype=self.ctx.torch_dtype, device=dev)
                     for C, dev in zip(num_chans, self.ntt.devices)]
                    for _ in range(2)]

        ksk_data = [k.data for k in ksk.data]

        csc.create_switcher_cuda(a, ksk_data, switcher, self.plan_ids[level], self.streams_id)

        ordinary_num_chans = [len(da) for da in self.ntt.p.destination_arrays[level]]
        switcher = [[s[:C] for s, C in zip(swit, ordinary_num_chans)] for swit in switcher]

        return switcher

    def create_switcher(self, a: list[torch.Tensor], ksk: DataStruct, level, exit_ntt=False) -> tuple:
        # ksk parts allocation.
        ksk_alloc = self.parts_alloc[level]

        # Device lens and neighbor devices.
        len_devices = self.len_devices[level]
        neighbor_devices = self.neighbor_devices[level]

        num_parts = sum([len(alloc) for alloc in ksk_alloc])
        part_results = [
            [
                [[] for _ in range(len_devices)],
                [[] for _ in range(len_devices)]
            ] for _ in range(num_parts)
        ]

        # 1. Generate states.
        states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for part_id in range(len(self.ntt.p.p[level][src_device_id])):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                state = self.pre_extend(a,
                                        src_device_id,
                                        level,
                                        part_id,
                                        exit_ntt
                                        )
                states[storage_id] = state

        # 2. Copy to CPU.
        CPU_states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                alpha = len(part)
                CPU_state = self.ksk_buffers[src_device_id][part_id][:alpha]
                CPU_state.copy_(states[storage_id], non_blocking=True)
                CPU_states[storage_id] = CPU_state

        # 3. Continue on with the follow-ups on source devices
        for src_device_id in range(len_devices):
            for part_id in range(len(self.ntt.p.p[level][src_device_id])):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                state = states[storage_id]
                d0, d1 = self.switcher_later_part(state, ksk,
                                                  src_device_id,
                                                  src_device_id,
                                                  level, part_id)

                part_results[storage_id][0][src_device_id] = d0
                part_results[storage_id][1][src_device_id] = d1

        # 4. Copy onto neighbor GPUs the states
        CUDA_states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for j, dst_device_id in enumerate(
                    neighbor_devices[src_device_id]):
                for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    CPU_state = CPU_states[storage_id]
                    CUDA_states[storage_id] = CPU_state.cuda(
                        self.ntt.devices[dst_device_id], non_blocking=True)

        # 5. Synchronize
        # torch.cuda.synchronize()

        # 6. Do follow-ups on neighbors
        for src_device_id in range(len_devices):
            for j, dst_device_id in enumerate(
                    neighbor_devices[src_device_id]):
                for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    CUDA_state = CUDA_states[storage_id]
                    d0, d1 = self.switcher_later_part(CUDA_state,
                                                      ksk,
                                                      src_device_id,
                                                      dst_device_id,
                                                      level,
                                                      part_id)
                    part_results[storage_id][0][dst_device_id] = d0
                    part_results[storage_id][1][dst_device_id] = d1

        # 7. Sum up.
        summed0 = part_results[0][0]
        summed1 = part_results[0][1]

        for i in range(1, len(part_results)):
            summed0 = self.ntt.mont_add(
                summed0, part_results[i][0], level, -2)
            summed1 = self.ntt.mont_add(
                summed1, part_results[i][1], level, -2)

        # Rename summed's
        d0 = summed0
        d1 = summed1

        # intt to prepare for division by P.
        self.ntt.intt_exit_reduce(d0, level, -2)
        self.ntt.intt_exit_reduce(d1, level, -2)

        # 6. Divide by P.
        # This is actually done in successive order.
        # Rescale from the most outer prime channel.
        # Start from the special len and drop channels one by one.

        # Pre-montgomery enter the ordinary part.
        # Note that special prime channels remain intact.
        c0 = [d[:-self.ntt.num_special_primes] for d in d0]
        c1 = [d[:-self.ntt.num_special_primes] for d in d1]

        self.ntt.mont_enter(c0, level, -1)
        self.ntt.mont_enter(c1, level, -1)

        current_len = [len(d) for d in self.ntt.p.destination_arrays_with_special[level]]

        for P_ind in range(self.ntt.num_special_primes):
            PiRi = self.PiRs[level][P_ind]

            # Tile.
            P0 = [d[-1 - P_ind].repeat(current_len[di], 1) for di, d in enumerate(d0)]
            P1 = [d[-1 - P_ind].repeat(current_len[di], 1) for di, d in enumerate(d1)]

            # mont enter only the ordinary part.
            Q0 = [d[:-self.ntt.num_special_primes] for d in P0]
            Q1 = [d[:-self.ntt.num_special_primes] for d in P1]

            self.ntt.mont_enter(Q0, level, -1)
            self.ntt.mont_enter(Q1, level, -1)

            # subtract P0 and P1.
            # Note that by the consequence of the above mont_enter
            # ordinary parts will be in montgomery form,
            # while the special part remains plain.
            d0 = self.ntt.mont_sub(d0, P0, level, -2)
            d1 = self.ntt.mont_sub(d1, P1, level, -2)

            self.ntt.mont_enter_scalar(d0, PiRi, level, -2)
            self.ntt.mont_enter_scalar(d1, PiRi, level, -2)

            self.ntt.reduce_2q(d0, level, -1)
            self.ntt.reduce_2q(d1, level, -1)

        # Carve out again, since d0 and d1 are fresh new.
        c0 = [d[:-self.ntt.num_special_primes] for d in d0]
        c1 = [d[:-self.ntt.num_special_primes] for d in d1]

        # Exit the montgomery.
        self.ntt.mont_redc(c0, level, -1)
        self.ntt.mont_redc(c1, level, -1)

        self.ntt.reduce_2q(c0, level, -1)
        self.ntt.reduce_2q(c1, level, -1)

        # 7. Return
        return c0, c1

    def switcher_later_part(self,
                            state, ksk,
                            src_device_id,
                            dst_device_id,
                            level, part_id):

        # Extend basis.
        extended = self.extend(
            state, src_device_id,
            level, part_id, dst_device_id)

        # ntt extended to prepare polynomial multiplication.
        # extended is in the Montgomery format already.
        self.ntt.ntt(
            [extended], level, dst_device_id, -2)

        # Extract the ksk.
        ksk_loc = self.parts_alloc[level][src_device_id][part_id]
        ksk_part_data = ksk.data[ksk_loc].data

        start = self.ntt.starts[level][dst_device_id]
        ksk0_data = ksk_part_data[0][dst_device_id][start:]
        ksk1_data = ksk_part_data[1][dst_device_id][start:]

        # Multiply.
        d0 = self.ntt.mont_mult(
            [extended], [ksk0_data], level, dst_device_id, -2)
        d1 = self.ntt.mont_mult(
            [extended], [ksk1_data], level, dst_device_id, -2)

        ## intt to prepare for division by P.
        # self.ntt.intt_exit_reduce(d0, level, dst_device_id, -2)
        # self.ntt.intt_exit_reduce(d1, level, dst_device_id, -2)

        # When returning, un-list the results by taking the 0th element.
        return d0[0], d1[0]

    def switch_key(self, ct: DataStruct, ksk: DataStruct, is_fast:bool=True) -> DataStruct:
        include_special = ct.include_special
        ntt_state = ct.ntt_state
        montgomery_state = ct.montgomery_state
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])

        level = ct.level_calc
        a = ct.data[1]
        if is_fast:
            d0, d1 = self.create_switcher_fast(a, ksk, level, exit_ntt=ct.ntt_state)
        else:
            d0, d1 = self.create_switcher(a, ksk, level, exit_ntt=ct.ntt_state)

        new_ct0 = self.ntt.mont_add(ct.data[0], d0, level, -1)
        self.ntt.reduce_2q(new_ct0, level, -1)

        return DataStruct(
            data=(new_ct0, d1),
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            origin=types.origins["ct"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    # -------------------------------------------------------------------------------------------
    # Bootstrapping.
    # -------------------------------------------------------------------------------------------
    def mult_ksk(self, extended, ksk: DataStruct, src_device_id, dst_device_id, level, part_id):
        # Extract the ksk.
        ksk_loc = self.parts_alloc[level][src_device_id][part_id]
        ksk_part_data = ksk.data[ksk_loc].data

        start = self.ntt.starts[level][dst_device_id]
        ksk0_data = ksk_part_data[0][dst_device_id][start:]
        ksk1_data = ksk_part_data[1][dst_device_id][start:]

        # Multiply.
        d0 = self.ntt.mont_mult(
            [extended], [ksk0_data], level, dst_device_id, -2)
        d1 = self.ntt.mont_mult(
            [extended], [ksk1_data], level, dst_device_id, -2)

        # intt to prepare for division by P.
        # self.ntt.intt_exit_reduce(d0, level, dst_device_id, -2)
        # self.ntt.intt_exit_reduce(d1, level, dst_device_id, -2)

        # When returning, un-list the results by taking the 0th element.
        return d0[0], d1[0]

    def decompose(self, a: list[torch.Tensor], level, exit_ntt=False) -> list:
        # ksk parts allocation.
        ksk_alloc = self.parts_alloc[level]

        # Device lens and neighbor devices.
        len_devices = self.len_devices[level]
        neighbor_devices = self.neighbor_devices[level]

        # Iterate over source device ids, and then part ids.
        num_parts = sum([len(alloc) for alloc in ksk_alloc])
        extended_part_result = [
            [
                [
                    [] for _ in range(len_devices)
                ] for _ in range(len_devices)
            ] for _ in range(num_parts)
        ]

        # 1. Generate states.
        states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for part_id in range(len(self.ntt.p.p[level][src_device_id])):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                state = self.pre_extend(
                    a,
                    src_device_id,
                    level,
                    part_id,
                    exit_ntt
                )
                states[storage_id] = state

        # 2. Copy to CPU.
        CPU_states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                alpha = len(part)
                CPU_state = self.ksk_buffers[src_device_id][part_id][:alpha]
                CPU_state.copy_(states[storage_id], non_blocking=True)
                CPU_states[storage_id] = CPU_state

        # 3. Continue on with the follow-ups on source devices.
        for src_device_id in range(len_devices):
            for part_id in range(len(self.ntt.p.p[level][src_device_id])):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                state = states[storage_id]
                # Extend basis.
                extended = self.extend(
                    state, src_device_id,
                    level, part_id, src_device_id)
                # ntt extended to prepare polynomial multiplication.
                # extended is in the Montgomery format already.
                self.ntt.ntt(
                    [extended], level, src_device_id, -2)
                extended_part_result[storage_id][src_device_id][src_device_id] = extended

        # 4. Copy onto neighbor GPUs the states.
        CUDA_states = [[] for _ in range(num_parts)]
        for src_device_id in range(len_devices):
            for j, dst_device_id in enumerate(
                    neighbor_devices[src_device_id]):
                for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    CPU_state = CPU_states[storage_id]
                    CUDA_states[storage_id] = CPU_state.cuda(
                        self.ntt.devices[dst_device_id], non_blocking=True)

        # 5. Synchronize
        # torch.cuda.synchronize()

        # 6. Do follow-ups on neighbors.
        for src_device_id in range(len_devices):
            for j, dst_device_id in enumerate(
                    neighbor_devices[src_device_id]):
                for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    CUDA_state = CUDA_states[storage_id]
                    # Extend basis.
                    extended = self.extend(
                        CUDA_state, src_device_id,
                        level, part_id, dst_device_id)
                    # ntt extended to prepare polynomial multiplication.
                    # extended is in the Montgomery format already.
                    self.ntt.ntt(
                        [extended], level, dst_device_id, -2)
                    extended_part_result[storage_id][src_device_id][dst_device_id] = extended

        return extended_part_result

    def mult_sum(self, decomposed_part_results, ksk, level):
        # ksk parts allocation.
        ksk_alloc = self.parts_alloc[level]

        # Device lens and neighbor devices.
        len_devices = self.len_devices[level]
        neighbor_devices = self.neighbor_devices[level]

        # Iterate over source device ids, and then part ids.
        num_parts = sum([len(alloc) for alloc in ksk_alloc])
        part_results = [
            [
                [[] for _ in range(len_devices)],
                [[] for _ in range(len_devices)]
            ]
            for _ in range(num_parts)
        ]

        for src_device_id in range(len_devices):
            for part_id, _ in enumerate(self.ntt.p.p[level][src_device_id]):
                storage_id = self.stor_ids[level][src_device_id][part_id]
                d0, d1 = self.mult_ksk(decomposed_part_results[storage_id][src_device_id][src_device_id],
                                       ksk,
                                       src_device_id,
                                       src_device_id,
                                       level, part_id)
                part_results[storage_id][0][src_device_id] = d0
                part_results[storage_id][1][src_device_id] = d1

        for src_device_id in range(len_devices):
            for _, dst_device_id in enumerate(
                    neighbor_devices[src_device_id]):
                for part_id, part in enumerate(self.ntt.p.p[level][src_device_id]):
                    storage_id = self.stor_ids[level][src_device_id][part_id]
                    d0, d1 = self.mult_ksk(decomposed_part_results[storage_id][src_device_id][dst_device_id],
                                           ksk,
                                           src_device_id,
                                           dst_device_id,
                                           level,
                                           part_id)
                    part_results[storage_id][0][dst_device_id] = d0
                    part_results[storage_id][1][dst_device_id] = d1

        # 7. Sum up.
        summed0 = part_results[0][0]
        summed1 = part_results[0][1]

        for i in range(1, len(part_results)):
            summed0 = self.ntt.mont_add(
                summed0, part_results[i][0], level, -2)
            summed1 = self.ntt.mont_add(
                summed1, part_results[i][1], level, -2)

        return summed0, summed1

    def mod_down(self, summed0, level):
        d0 = summed0

        c0 = [d[:-self.ntt.num_special_primes] for d in d0]
        self.ntt.mont_enter(c0, level, -1)

        current_len = [len(d) for d in self.ntt.p.destination_arrays_with_special[level]]

        for P_ind in range(self.ntt.num_special_primes):
            PiRi = self.PiRs[level][P_ind]

            # Tile.
            P0 = [d[-1 - P_ind].repeat(current_len[di], 1) for di, d in enumerate(d0)]

            # mont enter only the ordinary part.
            Q0 = [d[:-self.ntt.num_special_primes] for d in P0]

            self.ntt.mont_enter(Q0, level, -1)

            # subtract P0 and P1.
            # Note that by the consequence of the above mont_enter
            # ordinary parts will be in montgomery form,
            # while the special part remains plain.
            d0 = self.ntt.mont_sub(d0, P0, level, -2)

            self.ntt.mont_enter_scalar(d0, PiRi, level, -2)

        # Carve out again, since d0 and d1 are fresh new.
        c0 = [d[:-self.ntt.num_special_primes] for d in d0]

        # Exit the montgomery.
        self.ntt.mont_redc(c0, level, -1)
        self.ntt.reduce_2q(c0, level, -1)

        # 7. Return
        return c0

    def rotate_hoisting(self, ct: DataStruct, rotk: DataStruct) -> DataStruct:
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

        self.ntt.mont_enter_scalar(c0, self.mont_PR, level)
        self.ntt.mont_enter_scalar(c1, self.mont_PR, level)

        c0 = [torch.cat([c0[device_id],
                         torch.zeros([self.ntt.num_special_primes, self.ctx.N], device=self.ntt.devices[device_id],
                                     dtype=self.ctx.torch_dtype)], dim=0) for device_id in range(self.ntt.num_devices)]

        summed0, summed1 = self.mult_sum(decomposed_part_results, rotk, level)
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
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    # -------------------------------------------------------------------------------------------
    # Multiplication.
    # -------------------------------------------------------------------------------------------

    def rescale(self, ct: DataStruct, exact_rounding=True) -> DataStruct:
        data_count = len(ct.data)
        level = ct.level_calc
        next_level = level + 1
    
        if next_level >= self.num_levels:
            raise errors.MaximumLevelError(level=ct.level_calc, level_max=self.num_levels)
    
        rescaler_device_id = self.ntt.p.rescaler_loc[level]
        neighbor_devices_before = self.neighbor_devices[level]
        neighbor_devices_after = self.neighbor_devices[next_level]
        len_devices_after = len(neighbor_devices_after)
        len_devices_before = len(neighbor_devices_before)
    
        datas = []
    
        for data_id in range(data_count):
            if ct.origin == types.origins["ctt"]:
                self.ntt.intt_exit_reduce(ct.data[data_id], level)
            data = [[] for _ in range(len_devices_after)]
            rescaler = [[] for _ in range(len_devices_before)]
            rescaler_at = ct.data[data_id][rescaler_device_id][0]
    
            rescaler[rescaler_device_id] = rescaler_at
            if rescaler_device_id < len_devices_after:
                data[rescaler_device_id] = ct.data[data_id][rescaler_device_id][1:]
    
            CPU_rescaler = self.ksk_buffers[0][data_id][0] 
            CPU_rescaler.copy_(rescaler_at, non_blocking=False)
    
            for device_id in neighbor_devices_before[rescaler_device_id]:
                device = self.ntt.devices[device_id]
                CUDA_rescaler = CPU_rescaler.cuda(device)
                rescaler[device_id] = CUDA_rescaler
                if device_id < len_devices_after:
                    data[device_id] = ct.data[data_id][device_id]
    
            if exact_rounding:
                rescale_channel_prime_id = self.ntt.p.destination_arrays[level][rescaler_device_id][0]
                round_at = self.ctx.q[rescale_channel_prime_id] // 2
                rounder = [[] for _ in range(len_devices_before)]
                for device_id in range(len_devices_after):
                    rounder[device_id] = torch.where(rescaler[device_id] > round_at, 1, 0)
    
            data = [(d - s) for d, s in zip(data, rescaler)]
            self.ntt.mont_enter_scalar(data, self.rescale_scales[level], next_level)
    
            if exact_rounding:
                data = [(d + r) for d, r in zip(data, rounder)]
    
            self.ntt.reduce_2q(data, next_level)
            datas.append(data)

            if ct.origin == types.origins["ctt"]:
                self.ntt.enter_ntt(ct.data[data_id], level)
                self.ntt.enter_ntt(datas[data_id], next_level)
    
        return DataStruct(
            data=datas,
            include_special=False,
            ntt_state=ct.ntt_state,
            montgomery_state=ct.montgomery_state,
            origin=ct.origin,
            level_calc=next_level,
            level_available=self.num_levels,
            hash=self.hash,
            version=self.version
        )

    def rescale_bootstrapping(self, ct: DataStruct, exact_rounding=True) -> DataStruct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        level = ct.level_calc
        next_level = level + 1

        if next_level > self.num_levels:
            raise errors.MaximumLevelError(level=ct.level_calc, level_max=self.num_levels)

        rescaler_device_id = self.ntt.p.rescaler_loc[level]
        neighbor_devices_before = self.neighbor_devices[level]
        # TODO: multi-gpu apply
        neighbor_devices_after = self.neighbor_devices[level]
        len_devices_after = len(neighbor_devices_after)
        len_devices_before = len(neighbor_devices_before)

        data0 = [[] for _ in range(len_devices_after)]
        data1 = [[] for _ in range(len_devices_after)]

        rescaler0 = [[] for _ in range(len_devices_before)]
        rescaler1 = [[] for _ in range(len_devices_before)]

        rescaler0_at = ct.data[0][rescaler_device_id][0]
        rescaler0[rescaler_device_id] = rescaler0_at

        rescaler1_at = ct.data[1][rescaler_device_id][0]
        rescaler1[rescaler_device_id] = rescaler1_at

        if rescaler_device_id < len_devices_after:
            data0[rescaler_device_id] = ct.data[0][rescaler_device_id][1:]
            data1[rescaler_device_id] = ct.data[1][rescaler_device_id][1:]

        CPU_rescaler0 = self.ksk_buffers[0][0][0]
        CPU_rescaler1 = self.ksk_buffers[0][1][0]

        CPU_rescaler0.copy_(rescaler0_at, non_blocking=False)
        CPU_rescaler1.copy_(rescaler1_at, non_blocking=False)

        for device_id in neighbor_devices_before[rescaler_device_id]:
            device = self.ntt.devices[device_id]
            CUDA_rescaler0 = CPU_rescaler0.cuda(device)
            CUDA_rescaler1 = CPU_rescaler1.cuda(device)

            rescaler0[device_id] = CUDA_rescaler0
            rescaler1[device_id] = CUDA_rescaler1

            if device_id < len_devices_after:
                data0[device_id] = ct.data[0][device_id]
                data1[device_id] = ct.data[1][device_id]

        if exact_rounding:
            rescale_channel_prime_id = self.ntt.p.destination_arrays[level][rescaler_device_id][0]

            round_at = self.ctx.q[rescale_channel_prime_id] // 2

            rounder0 = [[] for _ in range(len_devices_before)]
            rounder1 = [[] for _ in range(len_devices_before)]

            for device_id in range(len_devices_after):
                rounder0[device_id] = torch.where(rescaler0[device_id] > round_at, 1, 0)
                rounder1[device_id] = torch.where(rescaler1[device_id] > round_at, 1, 0)

        data0 = [(d - s) for d, s in zip(data0, rescaler0)]
        data1 = [(d - s) for d, s in zip(data1, rescaler1)]

        self.ntt.mont_enter_scalar(data0, self.rescale_scales[level], next_level)

        self.ntt.mont_enter_scalar(data1, self.rescale_scales[level], next_level)

        if exact_rounding:
            data0 = [(d + r) for d, r in zip(data0, rounder0)]
            data1 = [(d + r) for d, r in zip(data1, rounder1)]

        self.ntt.reduce_2q(data0, next_level)
        self.ntt.reduce_2q(data1, next_level)

        return DataStruct(
            data=(data0, data1),
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level_calc=next_level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    def create_evk(self, sk: DataStruct) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        sk2_data = self.ntt.mont_mult(sk.data, sk.data, 0, -2)
        sk2 = DataStruct(
            data=sk2_data,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level_calc=sk.level_calc,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

        return self.create_key_switching_key(sk2, sk)

    def __cc_mult(self,
                  a: DataStruct, b: DataStruct,
                  evk: DataStruct,
                  relin: bool = True, rescale: bool = True) -> DataStruct:
        if a.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=a.origin, to=types.origins["sk"])
        if b.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=b.origin, to=types.origins["sk"])
        
        x = self.clone(a)
        y = self.clone(b)
        
        if rescale:
            # Rescale.
            x = self.rescale(x)
            y = self.rescale(y)

        level = x.level_calc

        # Multiply.
        a0 = x.data[0]
        a1 = x.data[1]

        b0 = y.data[0]
        b1 = y.data[1]

        self.ntt.enter_ntt(a0, level)
        self.ntt.enter_ntt(a1, level)
        self.ntt.enter_ntt(b0, level)
        self.ntt.enter_ntt(b1, level)

        d0 = self.ntt.mont_mult(a0, b0, level)

        a0b1 = self.ntt.mont_mult(a0, b1, level)
        a1b0 = self.ntt.mont_mult(a1, b0, level)
        d1 = self.ntt.mont_add(a0b1, a1b0, level)

        d2 = self.ntt.mont_mult(a1, b1, level)

        ct_mult = DataStruct(
            data=(d0, d1, d2),
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ctt"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash)
        )
        if relin:
            ct_mult = self.relinearize(ct_triplet=ct_mult, evk=evk)

        return ct_mult

    def cc_mult(self,
                ct_a: DataStruct, ct_b: DataStruct,
                evk: DataStruct,
                relin: bool = True, rescale: bool = True) -> DataStruct:
        """
            Perform homomorphic multiplication of two ciphertexts.

            This function takes two ciphertexts, `a` and `b`, and performs homomorphic
            multiplication on them. Optionally, the result can be re-linearized and rescaled
            depending on the `relin` and `rescale` parameters.
        @param ct_a: The first ciphertext to be multiplied.
        @param ct_b: The second ciphertext to be multiplied.
        @param evk: The evaluation key used for relinearization.
        @param relin: Flag indicating whether to perform relinearization after multiplication.
        @param rescale: Flag indicating whether to perform rescaling before multiplication.
        @return:
        """
        if ct_a is ct_b:
            ct_mult = self.square(ct=ct_a, evk=evk, relin=False, rescale=rescale)
        else:
            ct_mult = self.__cc_mult(a=ct_a, b=ct_b, evk=evk, relin=False, rescale=rescale)

        if relin:
            ct_mult = self.relinearize(ct_triplet=ct_mult, evk=evk)

        return ct_mult

    def relinearize(self, ct_triplet: DataStruct, evk: DataStruct, is_fast=True) -> DataStruct:
        if ct_triplet.origin != types.origins["ctt"]:
            raise errors.NotMatchType(origin=ct_triplet.origin, to=types.origins["ctt"])
        if not ct_triplet.ntt_state or not ct_triplet.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct_triplet.origin)

        d0, d1, d2 = ct_triplet.data
        level = ct_triplet.level_calc

        # intt.
        self.ntt.intt_exit_reduce(d0, level)
        self.ntt.intt_exit_reduce(d1, level)
        self.ntt.intt_exit_reduce(d2, level)

        # Key switch the x1y1.
        if is_fast:
            d2_0, d2_1 = self.create_switcher_fast(d2, evk, level)
        else:
            d2_0, d2_1 = self.create_switcher(d2, evk, level)

        # Add the switcher to d0, d1.
        d0 = [p + q for p, q in zip(d0, d2_0)]
        d1 = [p + q for p, q in zip(d1, d2_1)]

        # Final reduction.
        self.ntt.reduce_2q(d0, level)
        self.ntt.reduce_2q(d1, level)

        # Compose and return.
        return DataStruct(
            data=(d0, d1),
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash)
        )

    # -------------------------------------------------------------------------------------------
    # Rotation.
    # -------------------------------------------------------------------------------------------

    def create_rotation_key(self, sk: DataStruct, delta: int, a: list[torch.Tensor] = None) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [rotate(s, delta) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = DataStruct(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level_calc=0,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

        rotk = self.create_key_switching_key(sk_rotated, sk, crs=a)
        rotk = rotk._replace(origin=types.origins["rotk"] + f"{delta}")
        return rotk

    def rotate_single(self, ct: DataStruct, rotk: DataStruct, is_fast=True) -> DataStruct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if types.origins["rotk"] not in rotk.origin:
            raise errors.NotMatchType(origin=rotk.origin, to=types.origins["rotk"])

        level = ct.level_calc
        include_special = ct.include_special
        ntt_state = ct.ntt_state
        montgomery_state = ct.montgomery_state
        origin = rotk.origin
        delta = int(origin.split(':')[-1])

        rotated_ct_data = [[rotate(d, delta) for d in ct_data] for ct_data in ct.data]
        mult_type = -2 if include_special else -1
        for ct_data in rotated_ct_data:
            self.ntt.make_unsigned(ct_data, level, mult_type)
            self.ntt.reduce_2q(ct_data, level, mult_type)

        # Rotated ct may contain negative numbers.
        mult_type = -2 if include_special else -1
        for ct_data in rotated_ct_data:
            self.ntt.make_unsigned(ct_data, level, mult_type)
            self.ntt.reduce_2q(ct_data, level, mult_type)

        rotated_ct_rotated_sk = DataStruct(
            data=rotated_ct_data,
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            origin=types.origins["ct"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version,
        )

        rotated_ct = self.switch_key(rotated_ct_rotated_sk, rotk, is_fast=is_fast)
        return rotated_ct

    def create_galois_key(self, sk: DataStruct) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        galois_key_parts = [self.create_rotation_key(sk, delta) for delta in self.galois_deltas]

        galois_key = DataStruct(
            data=galois_key_parts,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["galk"],
            level_calc=0,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version,
        )
        return galois_key

    def rotate_galois(self, ct: DataStruct, gk: DataStruct, delta: int, return_circuit=False,
                      is_fast=True) -> DataStruct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if gk.origin != types.origins["galk"]:
            raise errors.NotMatchType(origin=gk.origin, to=types.origins["galk"])

        current_delta = delta % (self.ctx.N // 2)
        galois_circuit = []

        while current_delta:
            galois_ind = int(math.log2(current_delta))
            galois_delta = self.galois_deltas[galois_ind]
            galois_circuit.append(galois_ind)
            current_delta -= galois_delta

        if len(galois_circuit) > 0:
            rotated_ct = self.rotate_single(ct, gk.data[galois_circuit[0]], is_fast=is_fast)

            for delta_ind in galois_circuit[1:]:
                rotated_ct = self.rotate_single(rotated_ct, gk.data[delta_ind], is_fast=is_fast)
        elif len(galois_circuit) == 0:
            rotated_ct = ct
        else:
            pass

        if return_circuit:
            return rotated_ct, galois_circuit
        else:
            return rotated_ct

    def rotate_ct(self, ct: DataStruct, delta: int) -> DataStruct:
        level = ct.level_calc
        include_special = ct.include_special
        ntt_state = ct.ntt_state
        montgomery_state = ct.montgomery_state
        ct_rotated_data = [[rotate(d, delta) for d in ct_data] for ct_data in ct.data]

        mult_type = -2 if include_special else -1
        for ct_data in ct_rotated_data:
            self.ntt.make_unsigned(ct_data, level, mult_type)
            self.ntt.reduce_2q(ct_data, level, mult_type)

        ct_rotated = DataStruct(
            data=ct_rotated_data,
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            origin=types.origins["ct"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

        return ct_rotated

    def rotate_key(self, sk: DataStruct, delta: int) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        include_special = True
        mult_type = -2 if include_special else -1

        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data, 0, mult_type)
        sk_new_data = [rotate(s, delta) for s in sk_new_data]
        self.ntt.ntt(sk_new_data, 0, mult_type)
        sk_rotated = DataStruct(
            data=sk_new_data,
            include_special=include_special,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level_calc=0,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

        return sk_rotated

    def create_hoisting_rotation_key(self, sk: DataStruct, delta: int, a: list[torch.Tensor] = None) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])

        sk_copy = DataStruct(
            data=[s.clone() for s in sk.data],
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level_calc=0,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

        sk_rotated = self.rotate_key(sk, -delta)
        rotk = self.create_key_switching_key(sk_copy, sk_rotated, crs=a)
        rotk = rotk._replace(origin=types.origins["rotk"] + f"{delta}")

        return rotk

    # -------------------------------------------------------------------------------------------
    # Add/sub.
    # -------------------------------------------------------------------------------------------
    def cc_add_double(self, a: DataStruct, b: DataStruct) -> DataStruct:
        if a.origin != types.origins["ct"] or b.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=f"{a.origin} and {b.origin}", to=types.origins["ct"])
        if a.ntt_state or a.montgomery_state:
            raise errors.NotMatchDataStructState(origin=a.origin)
        if b.ntt_state or b.montgomery_state:
            raise errors.NotMatchDataStructState(origin=b.origin)

        level = a.level_calc
        data = []
        c0 = self.ntt.mont_add(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_add(a.data[1], b.data[1], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        data.extend([c0, c1])

        return DataStruct(
            data=data,
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    def cc_add_triplet(self, a: DataStruct, b: DataStruct) -> DataStruct:
        if a.origin != types.origins["ctt"] or b.origin != types.origins["ctt"]:
            raise errors.NotMatchType(origin=f"{a.origin} and {b.origin}", to=types.origins["ctt"])
        if not a.ntt_state or not a.montgomery_state:
            raise errors.NotMatchDataStructState(origin=a.origin)
        if not b.ntt_state or not b.montgomery_state:
            raise errors.NotMatchDataStructState(origin=b.origin)

        level = a.level_calc
        data = []
        c0 = self.ntt.mont_add(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_add(a.data[1], b.data[1], level)
        c2 = self.ntt.mont_add(a.data[2], b.data[2], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        self.ntt.reduce_2q(c2, level)
        data.extend([c0, c1, c2])

        return DataStruct(
            data=data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ctt"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    def cc_add(self, a: DataStruct, b: DataStruct) -> DataStruct:
        if a.origin == types.origins["ct"] and b.origin == types.origins["ct"]:
            ct_add = self.cc_add_double(a, b)
        elif a.origin == types.origins["ctt"] and b.origin == types.origins["ctt"]:
            ct_add = self.cc_add_triplet(a, b)
        else:
            raise errors.DifferentTypeError(a=a.origin, b=b.origin)

        return ct_add

    def cc_sub_double(self, a: DataStruct, b: DataStruct) -> DataStruct:
        if a.origin != types.origins["ct"] or b.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=f"{a.origin} and {b.origin}", to=types.origins["ct"])
        if a.ntt_state or a.montgomery_state:
            raise errors.NotMatchDataStructState(origin=a.origin)
        if b.ntt_state or b.montgomery_state:
            raise errors.NotMatchDataStructState(origin=b.origin)

        level = a.level_calc
        data = []

        c0 = self.ntt.mont_sub(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_sub(a.data[1], b.data[1], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        data.extend([c0, c1])

        return DataStruct(
            data=data,
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    def cc_sub_triplet(self, a: DataStruct, b: DataStruct) -> DataStruct:
        if a.origin != types.origins["ctt"] or b.origin != types.origins["ctt"]:
            raise errors.NotMatchType(origin=f"{a.origin} and {b.origin}", to=types.origins["ctt"])
        if not a.ntt_state or not a.montgomery_state:
            raise errors.NotMatchDataStructState(origin=a.origin)
        if not b.ntt_state or not b.montgomery_state:
            raise errors.NotMatchDataStructState(origin=b.origin)

        level = a.level_calc
        data = []
        c0 = self.ntt.mont_sub(a.data[0], b.data[0], level)
        c1 = self.ntt.mont_sub(a.data[1], b.data[1], level)
        c2 = self.ntt.mont_sub(a.data[2], b.data[2], level)
        self.ntt.reduce_2q(c0, level)
        self.ntt.reduce_2q(c1, level)
        self.ntt.reduce_2q(c2, level)
        data.extend([c0, c1, c2])

        return DataStruct(
            data=data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ctt"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    def cc_sub(self, a: DataStruct, b: DataStruct) -> DataStruct:
        if a.origin != b.origin:
            raise Exception(f"[Error] triplet error")

        if types.origins["ct"] == a.origin and types.origins["ct"] == b.origin:
            ct_sub = self.cc_sub_double(a, b)
        elif a.origin == types.origins["ctt"] and b.origin == types.origins["ctt"]:
            ct_sub = self.cc_sub_triplet(a, b)
        else:
            raise errors.DifferentTypeError(a=a.origin, b=b.origin)
        return ct_sub

    def cc_subtract(self, a, b):
        return self.cc_sub(a, b)

    # -------------------------------------------------------------------------------------------
    # Level up.
    # -------------------------------------------------------------------------------------------
    def __level_up(self, ct: DataStruct, dst_level: int) -> DataStruct:
        new_ct = self.rescale(ct)
        
        data_count = len(ct.data)

        src_level = new_ct.level_calc

        dst_len_devices = len(self.ntt.p.destination_arrays[dst_level])

        diff_deviation = self.deviations[dst_level] / np.sqrt(self.deviations[src_level])

        deviated_delta = round((2 ** round(math.log2(self.ctx.q[ct.level_calc]))) * diff_deviation)

        src_rns_lens = [len(d) for d in self.ntt.p.destination_arrays[src_level]]
        dst_rns_lens = [len(d) for d in self.ntt.p.destination_arrays[dst_level]]

        diff_rns_lens = [y - x for x, y in zip(dst_rns_lens, src_rns_lens)]
        
        new_datas = []
        
        for data_index in range(data_count):
            new_data = []
            for device_id in range(dst_len_devices):
                new_data.append(new_ct.data[data_index][device_id][diff_rns_lens[device_id]:])

            multipliers = []
            for device_id in range(dst_len_devices):
                dest = self.ntt.p.destination_arrays[dst_level][device_id]
                q = [self.ctx.q[i] for i in dest]

                multiplier = [(deviated_delta * self.ctx.R) % qi for qi in q]
                multiplier = torch.tensor(multiplier, dtype=self.ctx.torch_dtype, device=self.ntt.devices[device_id])
                multipliers.append(multiplier)

            self.ntt.mont_enter_scalar(new_data, multipliers, dst_level)
            self.ntt.reduce_2q(new_data, dst_level)
            
            new_datas.append(new_data)

        return DataStruct(
            data=new_datas,
            include_special=False,
            ntt_state=ct.ntt_state,
            montgomery_state=ct.montgomery_state,
            origin=ct.origin,
            level_calc=dst_level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )


    def level_up(self, ct: DataStruct, dst_level: int = None) -> DataStruct:
        """

        @param ct:
        @param dst_level:
        @return:
        """
        if dst_level is None:
            dst_level = ct.level_calc + 1

        current_level = ct.level_calc
        if current_level > dst_level:
            raise errors.LevelError(cipher_text_level=current_level, dst_level=dst_level)
        elif current_level == dst_level:
            return ct
        else:
            return self.__level_up(ct=ct, dst_level=dst_level)
    # -------------------------------------------------------------------------------------------
    # Fused enc/dec.
    # -------------------------------------------------------------------------------------------
    def encodecrypt(self, m, pk: DataStruct, level: int = 15, padding=True) -> DataStruct:
        if pk.origin != types.origins["pk"]:
            raise errors.NotMatchType(origin=pk.origin, to=types.origins["pk"])

        if padding:
            m = self.padding(m=m)

        deviation = self.deviations[level]
        pt = encode(m, scale=self.scale,
                    device=self.device0, norm=self.norm,
                    deviation=deviation, rng=self.rng,
                    return_without_scaling=self.bias_guard)

        if self.bias_guard:
            dc_integral = pt[0].item() // 1
            pt[0] -= dc_integral

            dc_scale = int(dc_integral) * int(self.scale)
            dc_rns = []
            for device_id, dest in enumerate(self.ntt.p.destination_arrays[level]):
                dci = [dc_scale % self.ctx.q[i] for i in dest]
                dci = torch.tensor(dci,
                                   dtype=self.ctx.torch_dtype,
                                   device=self.ntt.devices[device_id])
                dc_rns.append(dci)

            pt *= np.float64(self.scale)
            pt = self.rng.randround(pt)

        encoded = [pt]

        pt_buffer = self.ksk_buffers[0][0][0]
        pt_buffer.copy_(encoded[-1])
        for dev_id in range(1, self.ntt.num_devices):
            encoded.append(pt_buffer.cuda(self.ntt.devices[dev_id]))

        mult_type = -2 if pk.include_special else -1

        e0e1 = self.rng.discrete_gaussian(repeats=2)

        e0 = [e[0] for e in e0e1]
        e1 = [e[1] for e in e0e1]

        e0_tiled = self.ntt.tile_unsigned(e0, level, mult_type)
        e1_tiled = self.ntt.tile_unsigned(e1, level, mult_type)

        pt_tiled = self.ntt.tile_unsigned(encoded, level, mult_type)

        if self.bias_guard:
            for device_id, pti in enumerate(pt_tiled):
                pti[:, 0] += dc_rns[device_id]

        self.ntt.mont_enter_scale(pt_tiled, level, mult_type)
        self.ntt.mont_redc(pt_tiled, level, mult_type)
        pte0 = self.ntt.mont_add(pt_tiled, e0_tiled, level, mult_type)

        start = self.ntt.starts[level]
        pk0 = [pk.data[0][di][start[di]:] for di in range(self.ntt.num_devices)]
        pk1 = [pk.data[1][di][start[di]:] for di in range(self.ntt.num_devices)]

        v = self.rng.randint(amax=2, shift=0, repeats=1)

        v = self.ntt.tile_unsigned(v, level, mult_type)
        self.ntt.enter_ntt(v, level, mult_type)

        vpk0 = self.ntt.mont_mult(v, pk0, level, mult_type)
        vpk1 = self.ntt.mont_mult(v, pk1, level, mult_type)

        self.ntt.intt_exit(vpk0, level, mult_type)
        self.ntt.intt_exit(vpk1, level, mult_type)

        ct0 = self.ntt.mont_add(vpk0, pte0, level, mult_type)
        ct1 = self.ntt.mont_add(vpk1, e1_tiled, level, mult_type)

        self.ntt.reduce_2q(ct0, level, mult_type)
        self.ntt.reduce_2q(ct1, level, mult_type)

        ct = DataStruct(
            data=(ct0, ct1),
            include_special=mult_type == -2,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level_calc=level,
            level_available=self.num_levels-level,
            hash=self.strcp(self.hash),
            version=self.version
        )

        return ct

    def __decrode(self, ct: DataStruct, sk: DataStruct, is_real: bool = False, final_round: bool = True) -> np.ndarray:
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        level = ct.level_calc
        sk_data = sk.data[0][self.ntt.starts[level][0]:]

        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)

        ct0 = ct.data[0][0]
        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)

        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        pt = self.ntt.mont_add([ct0], sa, level)
        self.ntt.reduce_2q(pt, level)

        base_at = -self.ctx.num_special_primes - 1 if ct.include_special else -1
        base = pt[0][base_at][None, :]
        scaler = pt[0][0][None, :]

        len_left = len(self.ntt.p.destination_arrays[level][0])

        if (len_left >= 3) and self.bias_guard:
            dc0 = base[0][0].item()
            dc1 = scaler[0][0].item()
            dc2 = pt[0][1][0].item()

            base[0][0] = 0
            scaler[0][0] = 0

            q0_ind = self.ntt.p.destination_arrays[level][0][base_at]
            q1_ind = self.ntt.p.destination_arrays[level][0][0]
            q2_ind = self.ntt.p.destination_arrays[level][0][1]

            q0 = self.ctx.q[q0_ind]
            q1 = self.ctx.q[q1_ind]
            q2 = self.ctx.q[q2_ind]

            Q = q0 * q1 * q2
            Q0 = q1 * q2
            Q1 = q0 * q2
            Q2 = q0 * q1

            Qi0 = pow(Q0, -1, q0)
            Qi1 = pow(Q1, -1, q1)
            Qi2 = pow(Q2, -1, q2)

            dc = (dc0 * Qi0 * Q0 + dc1 * Qi1 * Q1 + dc2 * Qi2 * Q2) % Q

            half_Q = Q // 2
            dc = dc if dc <= half_Q else dc - Q

            dc = (dc + (q1 - 1)) // q1

        final_scalar = self.final_scalar[level]
        scaled = self.ntt.mont_sub([base], [scaler], -1)
        self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
        self.ntt.reduce_2q(scaled, -1)
        self.ntt.make_signed(scaled, -1)

        # Round?
        if final_round:
            # The scaler and the base channels are guaranteed to be in the
            # device 0.
            rounding_prime = self.ntt.qlists[0][-self.ctx.num_special_primes - 2]
            rounder = (scaler[0] > (rounding_prime // 2)) * 1
            scaled[0] += rounder

        # Decoding.
        correction = self.corrections[level]
        decoded = decode(
            scaled[0][-1],
            scale=self.scale,
            correction=correction,
            norm=self.norm,
            return_without_scaling=self.bias_guard
        )
        decoded = decoded[:self.ctx.N // 2].cpu().numpy()
        ##

        decoded = decoded / self.scale * correction

        # Bias guard.
        if (len_left >= 3) and self.bias_guard:
            decoded += dc / self.scale * correction
        if is_real:
            decoded = decoded.real
        return decoded

    def decryptcode(self, ct: DataStruct, sk: DataStruct, is_real=False, final_round=True) -> np.ndarray:
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        m = None
        if ct.origin == types.origins["ct"]:
            if self.bias_guard:
                m = self.__decrode(ct=ct, sk=sk, is_real=is_real, final_round=final_round)
            elif self.bias_guard is False:
                pt = self.decrypt(ct=ct, sk=sk, final_round=final_round)
                m = self.decode(m=pt, level=ct.level_calc, is_real=is_real)
            else:
                raise "Error"
        elif ct.origin == types.origins["ctt"]:
            pt = self.decrypt_triplet(ct_mult=ct, sk=sk, final_round=final_round)
            m = self.decode(m=pt, level=ct.level_calc, is_real=is_real)
        else:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])

        return m

    # -------------------------------------------------------------------------------------------
    # Shortcuts
    # -------------------------------------------------------------------------------------------

    def imult(self, ct: DataStruct, neg: bool = False):
        """
            The multiplication of the ciphertext by the imaginary unit.
        @param ct:
        @param neg:
        @return:
        """
        return self.mult_imag(ct=ct, neg=neg)

    def encorypt(self, m, pk: DataStruct, level: int = 0, padding=True):
        return self.encodecrypt(m, pk=pk, level=level, padding=padding)

    def decrode(self, ct: DataStruct, sk: DataStruct, is_real=False, final_round=True) -> np.ndarray:
        return self.decryptcode(ct=ct, sk=sk, is_real=is_real, final_round=final_round)

    # -------------------------------------------------------------------------------------------
    # Conjugation
    # -------------------------------------------------------------------------------------------
    def create_conjugation_key(self, sk: DataStruct) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if (not sk.ntt_state) or (not sk.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk.origin)

        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [conjugate(s) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = DataStruct(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level_calc=0,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )
        rotk = self.create_key_switching_key(sk_rotated, sk)
        rotk = rotk._replace(origin=types.origins["conjk"])
        return rotk

    def conjugate(self, ct: DataStruct, conjk: DataStruct, is_fast=True):
        level = ct.level_calc
        conj_ct_data = [[conjugate(d) for d in ct_data] for ct_data in ct.data]
        self.ntt.make_unsigned(conj_ct_data[0], level, mult_type=-2)
        self.ntt.reduce_2q(conj_ct_data[0], level, mult_type=-2)
        self.ntt.make_unsigned(conj_ct_data[1], level, mult_type=-2)
        self.ntt.reduce_2q(conj_ct_data[1], level, mult_type=-2)

        conj_ct_sk = DataStruct(
            data=conj_ct_data,
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

        return self.switch_key(conj_ct_sk, conjk, is_fast=is_fast)

    # -------------------------------------------------------------------------------------------
    # Clone.
    # -------------------------------------------------------------------------------------------

    def clone_tensors(self, data: DataStruct) -> DataStruct:
        new_data = []
        # Some data has 1 depth.
        if not isinstance(data[0], list):
            for device_data in data:
                new_data.append(device_data.clone())
        else:
            for part in data:
                new_data.append([])
                for device_data in part:
                    new_data[-1].append(device_data.clone())
        return new_data

    def clone(self, text):
        if not isinstance(text.data[0], DataStruct):
            # data are tensors.
            data = self.clone_tensors(text.data)

            wrapper = DataStruct(
                data=data,
                include_special=text.include_special,
                ntt_state=text.ntt_state,
                montgomery_state=text.montgomery_state,
                origin=text.origin,
                level_calc=text.level_calc,
                level_available=text.level_available,
                hash=text.hash,
                version=text.version
            )

        else:
            wrapper = DataStruct(
                data=[],
                include_special=text.include_special,
                ntt_state=text.ntt_state,
                montgomery_state=text.montgomery_state,
                origin=text.origin,
                level_calc=text.level_calc,
                level_available=text.level_available,
                hash=text.hash,
                version=text.version
            )

            for d in text.data:
                wrapper.data.append(self.clone(d))

        return wrapper

    # -------------------------------------------------------------------------------------------
    # Move data back and forth from GPUs to the CPU.
    # -------------------------------------------------------------------------------------------
    def download_to_cpu(self, gpu_data, level, include_special):
        # Prepare destination arrays.
        if include_special:
            dest = self.ntt.p.destination_arrays_with_special[level]
        else:
            dest = self.ntt.p.destination_arrays[level]

        # dest contain indices that are the absolute order of primes.
        # Convert them to tensor channel indices at this level.
        # That is, force them to start from zero.
        min_ind = min([min(d) for d in dest])
        dest = [[di - min_ind for di in d] for d in dest]

        # Tensor size parameters.
        num_rows = sum([len(d) for d in dest])
        num_cols = self.ctx.N
        cpu_size = (num_rows, num_cols)

        # Make a cpu tensor to aggregate the data in GPUs.
        cpu_tensor = torch.empty(cpu_size, dtype=self.ctx.torch_dtype, device='cpu')

        for ten, dest_i in zip(gpu_data, dest):
            # Check if the tensor is in the gpu.
            if ten.device.type != 'cuda':
                raise Exception("To download data to the CPU, it must already be in a GPU!!!")

            # Copy in the data.
            cpu_tensor[dest_i] = ten.cpu()

        # To avoid confusion, make a list with a single element (only one device, that is the CPU),
        # and return it.
        return [cpu_tensor]

    def upload_to_gpu(self, cpu_data, level, include_special):
        # There's only one device data in the cpu data.
        cpu_tensor = cpu_data[0]

        # Check if the tensor is in the cpu.
        if cpu_tensor.device.type != 'cpu':
            raise Exception("To upload data to GPUs, it must already be in the CPU!!!")

        # Prepare destination arrays.
        if include_special:
            dest = self.ntt.p.destination_arrays_with_special[level]
        else:
            dest = self.ntt.p.destination_arrays[level]

        # dest contain indices that are the absolute order of primes.
        # Convert them to tensor channel indices at this level.
        # That is, force them to start from zero.
        min_ind = min([min(d) for d in dest])
        dest = [[di - min_ind for di in d] for d in dest]

        gpu_data = []
        for device_id in range(len(dest)):
            # Copy in the data.
            dest_device = dest[device_id]
            device = self.ntt.devices[device_id]
            gpu_tensor = cpu_tensor[dest_device].to(device=device)

            # Append to the gpu_data list.
            gpu_data.append(gpu_tensor)

        return gpu_data

    def move_tensors(self, data, level, include_special, direction):
        func = {
            'gpu2cpu': self.download_to_cpu,
            'cpu2gpu': self.upload_to_gpu
        }[direction]

        # Some data has 1 depth.
        if not isinstance(data[0], list):
            moved = func(data, level, include_special)
            new_data = moved
        else:
            new_data = []
            for part in data:
                moved = func(part, level, include_special)
                new_data.append(moved)
        return new_data

    def move_to(self, text, direction='gpu2cpu') -> DataStruct:
        if not isinstance(text.data[0], DataStruct):
            level = text.level_calc
            include_special = text.include_special

            # data are tensors.
            data = self.move_tensors(text.data, level,
                                     include_special, direction)

            wrapper = DataStruct(
                data=data,
                include_special=text.include_special,
                ntt_state=text.ntt_state,
                montgomery_state=text.montgomery_state,
                origin=text.origin,
                level_calc=text.level_calc,
                level_available=text.level_available,
                hash=text.hash,
                version=text.version
            )

        else:
            wrapper = DataStruct(
                data=[],
                include_special=text.include_special,
                ntt_state=text.ntt_state,
                montgomery_state=text.montgomery_state,
                origin=text.origin,
                level_calc=text.level_calc,
                level_available=text.level_available,
                hash=text.hash,
                version=text.version
            )

            for d in text.data:
                moved = self.move_to(d, direction)
                wrapper.data.append(moved)

        return wrapper

        # Shortcuts

    def cpu(self, ct):
        return self.move_to(ct, 'gpu2cpu')

    def cuda(self, ct):
        return self.move_to(ct, 'cpu2gpu')

    # -------------------------------------------------------------------------------------------
    # check device.
    # -------------------------------------------------------------------------------------------
    def tensor_device(self, data):
        # Some data has 1 depth.
        if not isinstance(data[0], list):
            return data[0].device.type
        else:
            return data[0][0].device.type

    def device(self, text):
        if not isinstance(text.data[0], DataStruct):
            # data are tensors.
            return self.tensor_device(text.data)
        else:
            return self.device(text.data[0])

    # -------------------------------------------------------------------------------------------
    # Print data structure.
    # -------------------------------------------------------------------------------------------
    def tree_lead_text(self, level, tabs=2, final=False):
        final_char = "└" if final else "├"

        if level == 0:
            leader = " " * tabs
            trailer = "─" * tabs
            lead_text = "─" * tabs + "┬" + trailer

        elif level < 0:
            level = -level
            leader = " " * tabs
            trailer = "─" + "─" * (tabs - 1)
            lead_fence = leader + "│" * (level - 1)
            lead_text = lead_fence + final_char + trailer

        else:
            leader = " " * tabs
            trailer = "┬" + "─" * (tabs - 1)
            lead_fence = leader + "│" * (level - 1)
            lead_text = lead_fence + "├" + trailer

        return lead_text

    def print_data_shapes(self, data, level):
        # Some data structures have 1 depth.
        if isinstance(data[0], list):
            for part_i, part in enumerate(data):
                for device_id, d in enumerate(part):
                    device = self.ntt.devices[device_id]

                    if (device_id == len(part) - 1) and \
                            (part_i == len(data) - 1):
                        final = True
                    else:
                        final = False

                    lead_text = self.tree_lead_text(-level, final=final)

                    print(f"{lead_text} tensor at device {device} with "
                          f"shape {d.shape}.")
        else:
            for device_id, d in enumerate(data):
                device = self.ntt.devices[device_id]

                if device_id == len(data) - 1:
                    final = True
                else:
                    final = False

                lead_text = self.tree_lead_text(-level, final=final)

                print(f"{lead_text} tensor at device {device} with "
                      f"shape {d.shape}.")

    def print_data_structure(self, text, level=0):
        lead_text = self.tree_lead_text(level)
        print(f"{lead_text} {text.origin}")

        if not isinstance(text.data[0], DataStruct):
            self.print_data_shapes(text.data, level + 1)
        else:
            for d in text.data:
                self.print_data_structure(d, level + 1)

    # -------------------------------------------------------------------------------------------
    # Save and load.
    # -------------------------------------------------------------------------------------------
    def auto_generate_filename(self, fmt_str='%Y%m%d%H%M%s%f'):
        return datetime.datetime.now().strftime(fmt_str) + '.pkl'

    def save(self, text, filename: str = None):
        if filename is None:
            filename = self.auto_generate_filename()

        savepath = Path(filename)

        # Check if the text is in the CPU.
        # If not, move to CPU.
        if self.device(text) != 'cpu':
            cpu_text = self.cpu(text)
        else:
            cpu_text = text

        with savepath.open('wb') as f:
            pickle.dump(cpu_text, f)

    def load(self, filename: str, move_to_gpu: bool = True):
        savepath = Path(filename)
        with savepath.open('rb') as f:
            # gc.disable()
            cpu_text = pickle.load(f)
            # gc.enable()

        if move_to_gpu:
            text = self.cuda(cpu_text)
        else:
            text = cpu_text

        return text

    # -------------------------------------------------------------------------------------------
    # Negate.
    # -------------------------------------------------------------------------------------------
    def negate(self, ct: DataStruct) -> DataStruct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])  # ctt
        new_ct = self.clone(ct)

        new_data = new_ct.data
        for part in new_data:
            for d in part:
                d *= -1
            self.ntt.make_signed(part, ct.level_calc)

        return new_ct

    # -------------------------------------------------------------------------------------------
    # scalar ops.
    # -------------------------------------------------------------------------------------------

    def mult_int_scalar(self, ct: DataStruct, scalar: int|np.int64,
                        evk: DataStruct = None,
                        relin: bool = True, rescale: bool = True)->DataStruct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])

        device_len = len(ct.data[0])

        int_scalar = int(scalar)
        mont_scalar = [(int_scalar * self.ctx.R) % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level_calc]

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
        for i in [0, 1]:
            self.ntt.mont_enter_scalar(new_data[i], tensorized_scalar, ct.level_calc)
            self.ntt.reduce_2q(new_data[i], ct.level_calc)

        return new_ct

    def mult_scalar(self, ct, scalar: float,
                    evk: DataStruct = None,
                    relin: bool = True, rescale: bool = True):
        device_len = len(ct.data[0])

        scaled_scalar = int(
            scalar * self.scale * np.sqrt(self.deviations[ct.level_calc + 1]) + 0.5)

        mont_scalar = [(scaled_scalar * self.ctx.R) % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level_calc]

        partitioned_mont_scalar = [[mont_scalar[i] for i in dest_i] for dest_i in dest]
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

        for i in [0, 1]:
            self.ntt.mont_enter_scalar(new_data[i], tensorized_scalar, ct.level_calc)
            self.ntt.reduce_2q(new_data[i], ct.level_calc)
        if rescale==True:
            return self.rescale(new_ct)
        else:
            return new_ct

    def mult_scalar_bootstrapping(self, ct, scalar: float,
                    evk: DataStruct = None,
                    relin: bool = True, rescale: bool = True):
        device_len = len(ct.data[0])

        scaled_scalar = int(
            scalar * float(2 ** 59) * np.sqrt(self.deviations[ct.level_calc + 1]) + 0.5)

        mont_scalar = [(scaled_scalar * self.ctx.R) % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level_calc]

        partitioned_mont_scalar = [[mont_scalar[i] for i in dest_i] for dest_i in dest]
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

        for i in [0, 1]:
            self.ntt.mont_enter_scalar(new_data[i], tensorized_scalar, ct.level_calc)
            self.ntt.reduce_2q(new_data[i], ct.level_calc)

        return self.rescale(new_ct)

    def add_scalar(self, ct, scalar):
        device_len = len(ct.data[0])

        scaled_scalar = int(scalar * self.scale * self.deviations[ct.level_calc] + 0.5)

        if self.norm == 'backward':
            scaled_scalar *= self.ctx.N

        scaled_scalar *= self.int_scale

        scaled_scalar = [scaled_scalar % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level_calc]

        partitioned_mont_scalar = [[scaled_scalar[i] for i in desti] for desti in dest]
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

        dc = [d[:, 0] for d in new_data[0]]
        for device_id in range(device_len):
            dc[device_id] += tensorized_scalar[device_id]

        self.ntt.reduce_2q(new_data[0], ct.level_calc)

        return new_ct

    def add_scalar_bootstrapping(self, ct, scalar):
        device_len = len(ct.data[0])

        scaled_scalar = int(scalar * float(2 ** 59) * self.deviations[ct.level_calc] + 0.5)

        if self.norm == 'backward':
            scaled_scalar *= self.ctx.N

        scaled_scalar *= 2 ** 59

        scaled_scalar = [scaled_scalar % qi for qi in self.ctx.q]

        dest = self.ntt.p.destination_arrays[ct.level_calc]

        partitioned_mont_scalar = [[scaled_scalar[i] for i in desti] for desti in dest]
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

        dc = [d[:, 0] for d in new_data[0]]
        for device_id in range(device_len):
            dc[device_id] += tensorized_scalar[device_id]

        self.ntt.reduce_2q(new_data[0], ct.level_calc)

        return new_ct

    def sub_scalar(self, ct: DataStruct, scalar: float | int):
        return self.add_scalar(ct=ct, scalar=-scalar)

    def int_scalar_mult(self,
                        scalar: int|np.int64, ct: DataStruct,
                        evk: DataStruct = None,
                        relin: bool = True, rescale: bool = True):
        return self.mult_int_scalar(ct=ct, scalar=scalar, evk=evk, relin=relin, rescale=rescale)

    def scalar_mult(self,
                    scalar: float, ct: DataStruct,
                    evk: DataStruct = None,
                    relin: bool = True, rescale: bool = True):
        return self.mult_scalar(ct, scalar)

    def scalar_add(self, scalar, ct):
        return self.add_scalar(ct, scalar)

    def scalar_sub(self, scalar, ct):
        neg_ct = self.negate(ct)
        return self.add_scalar(ct=neg_ct, scalar=scalar)

    # -------------------------------------------------------------------------------------------
    # message ops.
    # -------------------------------------------------------------------------------------------
    def mc_mult(self,
                m: np.ndarray | list, ct: DataStruct,
                evk: DataStruct = None,
                relin: bool = True, rescale: bool = True)->DataStruct:
        m = np.array(m) * np.sqrt(self.deviations[ct.level_calc + 1])

        pt = self.encode(m, 15)

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

        return self.rescale(new_ct)

    def mc_add(self, m, ct):
        pt = self.encode(m, ct.level_calc)
        pt_tiled = self.ntt.tile_unsigned(pt, ct.level_calc)

        self.ntt.mont_enter_scale(pt_tiled, ct.level_calc)

        new_ct = self.clone(ct)
        self.ntt.mont_enter(new_ct.data[0], ct.level_calc)
        new_d0 = self.ntt.mont_add(pt_tiled, new_ct.data[0], ct.level_calc)
        self.ntt.mont_redc(new_d0, ct.level_calc)
        self.ntt.reduce_2q(new_d0, ct.level_calc)

        new_ct.data[0] = new_d0

        return new_ct

    def mc_sub(self, m, ct):
        neg_ct = self.negate(ct)
        return self.mc_add(m, neg_ct)

    def cm_mult(self,
                ct: DataStruct, m: np.ndarray | list,
                evk: DataStruct = None,
                relin: bool = True, rescale: bool = True):
        return self.mc_mult(m=m, ct=ct, evk=evk, relin=relin, rescale=rescale)

    def cm_add(self, ct, m):
        return self.mc_add(m, ct)

    def cm_sub(self, ct, m):
        return self.mc_add(-np.array(m), ct)

    # -------------------------------------------------------------------------------------------
    # Automatic cc ops.
    # -------------------------------------------------------------------------------------------
    def auto_level(self, ct0:DataStruct, ct1:DataStruct):
        level_diff = ct0.level_calc - ct1.level_calc
        if level_diff < 0:
            new_ct0 = self.level_up(ct0, ct1.level_calc)
            return new_ct0, ct1
        elif level_diff > 0:
            new_ct1 = self.level_up(ct1, ct0.level_calc)
            return ct0, new_ct1
        else:
            return ct0, ct1

    def auto_cc_mult(self,
                     ct0: DataStruct, ct1: DataStruct,
                     evk: DataStruct,
                     relin: bool = True, rescale: bool = True):
        lct0, lct1 = self.auto_level(ct0, ct1)
        return self.cc_mult(lct0, lct1, evk, relin=relin, rescale=rescale)

    def auto_cc_add(self, ct0, ct1):
        lct0, lct1 = self.auto_level(ct0, ct1)
        return self.cc_add(lct0, lct1)

    def auto_cc_sub(self, ct0, ct1):
        lct0, lct1 = self.auto_level(ct0, ct1)
        return self.cc_sub(lct0, lct1)

    # -------------------------------------------------------------------------------------------
    # Fully automatic ops.
    # -------------------------------------------------------------------------------------------

    def mult(self,
             a: DataStruct | np.ndarray | list | float | int,
             b: DataStruct | np.ndarray | list | float | int,
             evk: DataStruct = None,
             relin: bool = True, rescale: bool = True):
        """

        @param a:
        @param b:
        @param evk:
        @param relin:
        @param rescale:
        @return:
        """
        type_a = type(a)
        type_b = type(b)

        try:
            func = self.dispatch_dict_mult[type_a, type_b]
        except Exception as e:
            raise Exception(f"Unsupported data types are input.\n{e}")

        return func(a, b, evk=evk, relin=relin, rescale=rescale)

    def mult_imag(self, ct: DataStruct, neg: bool = False):
        """
            The multiplication of the ciphertext by the imaginary unit.
        @param ct:
        @param neg:
        @return:
        """
        c = [torch.zeros(2 * self.num_slots, dtype=torch.int64, device=device) for device in self.ntt.devices]
        for i in range(self.ntt.num_devices):
            if neg:
                c[i][self.num_slots] = -1
            else:
                c[i][self.num_slots] = 1
        c_rns = self.ntt.tile_unsigned(c, ct.level_calc)
        self.ntt.enter_ntt(c_rns, ct.level_calc)
        
        temp = self.clone(ct)

        self.ntt.enter_ntt(temp.data[0], ct.level_calc)
        self.ntt.enter_ntt(temp.data[1], ct.level_calc)

        new_d0 = self.ntt.mont_mult(c_rns, temp.data[0], ct.level_calc)
        new_d1 = self.ntt.mont_mult(c_rns, temp.data[1], ct.level_calc)

        self.ntt.intt_exit_reduce(new_d0, ct.level_calc)
        self.ntt.intt_exit_reduce(new_d1, ct.level_calc)

        return DataStruct(
            data=(new_d0, new_d1),
            include_special=False,
            ntt_state=False,
            montgomery_state=False,
            origin=types.origins["ct"],
            level_calc=ct.level_calc,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    def add(self,
            a: DataStruct | int | float | list,
            b: DataStruct | int | float | list) -> DataStruct:
        """

        @param a:
        @param b:
        @return:
        """
        type_a = type(a)
        type_b = type(b)

        try:
            func = self.dispatch_dict_add[type_a, type_b]
        except Exception as e:
            raise Exception(f"Unsupported data types are input.\n{e}")

        return func(a, b)

    def sub(self,
            a: DataStruct | int | float | list,
            b: DataStruct | int | float | list) -> DataStruct:
        type_a = type(a)
        type_b = type(b)

        try:
            func = self.dispatch_dict_sub[type_a, type_b]
        except Exception as e:
            raise Exception(f"Unsupported data types are input.\n{e}")

        return func(a, b)

    # -------------------------------------------------------------------------------------------
    # Misc.
    # -------------------------------------------------------------------------------------------
    def refresh(self):
        # Refreshes the rng state.
        self.rng.refresh()

    def reduce_error(self, ct: DataStruct) -> DataStruct:
        # Reduce the accumulated error in the cipher text.
        return self.mult_scalar(ct, 1.0)

    # -------------------------------------------------------------------------------------------
    # Misc ops.
    # -------------------------------------------------------------------------------------------
    def sum(self, ct: DataStruct, gk):
        new_ct = self.clone(ct)
        for roti in range(self.ctx.logN - 1):
            rotk = gk.data[roti]
            rot_ct = self.rotate_single(new_ct, rotk)
            new_ct = self.add(rot_ct, new_ct)
        return new_ct

    def pow(self, ct: DataStruct, power: int, evk: DataStruct) -> DataStruct:
        current_exponent = 2
        pow_list = [ct]
        while current_exponent <= power:
            current_ct = pow_list[-1]
            new_ct = self.cc_mult(current_ct, current_ct, evk)
            pow_list.append(new_ct)
            current_exponent *= 2

        remaining_exponent = power - current_exponent // 2
        new_ct = pow_list[-1]

        while remaining_exponent > 0:
            pow_ind = math.floor(math.log2(remaining_exponent))
            pow_term = pow_list[pow_ind]
            new_ct = self.auto_cc_mult(new_ct, pow_term, evk)
            remaining_exponent -= 2 ** pow_ind

        return new_ct

    def square(self,
               ct: DataStruct,
               evk: DataStruct,
               relin=True, rescale: bool = True,
               is_fast: bool = True) -> DataStruct:
        if rescale:
            ct = self.rescale(ct)

        level = ct.level_calc

        # Multiply.
        x0, x1 = ct.data

        self.ntt.enter_ntt(x0, level)
        self.ntt.enter_ntt(x1, level)

        d0 = self.ntt.mont_mult(x0, x0, level)
        x0y1 = self.ntt.mont_mult(x0, x1, level)
        d1 = self.ntt.mont_add(x0y1, x0y1, level)

        d2 = self.ntt.mont_mult(x1, x1, level)

        ct_mult = DataStruct(
            data=(d0, d1, d2),
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ctt"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )
        if relin:
            ct_mult = self.relinearize(ct_triplet=ct_mult, evk=evk, is_fast=is_fast)

        return ct_mult

    # -------------------------------------------------------------------------------------------
    # Multiparty.
    # -------------------------------------------------------------------------------------------

    def multiparty_public_crs(self, pk: DataStruct) -> list[torch.tensor]:
        """
            Get public key's CRS.
        @param pk:
        @return:
        """
        crs = self.clone(pk).data[1]
        return crs

    def multiparty_create_public_key(self,
                                     sk: DataStruct,
                                     crs: list[torch.tensor]=None,
                                     include_special: bool = False) -> DataStruct:
        """

        @param sk:
        @param crs:
        @param include_special:
        @return:
        """
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if include_special and not sk.include_special:
            raise errors.SecretKeyNotIncludeSpecialPrime()
        mult_type = -2 if include_special else -1

        level = 0
        e = self.rng.discrete_gaussian(repeats=1)
        e = self.ntt.tile_unsigned(e, level, mult_type)

        self.ntt.enter_ntt(e, level, mult_type)
        repeats = self.ctx.num_special_primes if sk.include_special else 0

        if crs is None:
            crs = self.rng.randint(
                self.ntt.q_prepack[mult_type][level][0],
                repeats=repeats
            )

        sa = self.ntt.mont_mult(crs, sk.data, 0, mult_type)
        pk0 = self.ntt.mont_sub(e, sa, 0, mult_type)
        pk = DataStruct(
            data=(pk0, crs),
            include_special=include_special,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["pk"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )
        return pk

    def multiparty_create_collective_public_key(self, pks: list[DataStruct]) -> DataStruct:
        data, include_special, ntt_state, montgomery_state, origin, level, level_available, hash_, version = pks[0]
        mult_type = -2 if include_special else -1
        b = [b.clone() for b in data[0]]  # num of gpus
        a = [a.clone() for a in data[1]]

        for pk in pks[1:]:
            b = self.ntt.mont_add(b, pk.data[0], lvl=0, mult_type=mult_type)

        cpk = DataStruct(
            (b, a),
            include_special=include_special,
            ntt_state=ntt_state,
            montgomery_state=montgomery_state,
            origin=types.origins["pk"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )
        return cpk

    def multiparty_decrypt_head(self, ct: DataStruct, sk: DataStruct):
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)
        if not sk.ntt_state or not sk.montgomery_state:
            raise errors.NotMatchDataStructState(origin=sk.origin)
        level = ct.level_calc

        ct0 = ct.data[0][0]
        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)

        sk_data = sk.data[0][self.ntt.starts[level][0]:]

        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        pt = self.ntt.mont_add([ct0], sa, level)

        return pt

    def multiparty_decrypt_partial(self, ct: DataStruct, sk: DataStruct) -> DataStruct:
        if ct.origin != types.origins["ct"]:
            raise errors.NotMatchType(origin=ct.origin, to=types.origins["ct"])
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        if ct.ntt_state or ct.montgomery_state:
            raise errors.NotMatchDataStructState(origin=ct.origin)
        if not sk.ntt_state or not sk.montgomery_state:
            raise errors.NotMatchDataStructState(origin=sk.origin)

        data, include_special, ntt_state, montgomery_state, origin, level, level_available, hash_, version = ct

        a = ct.data[1][0].clone()

        self.ntt.enter_ntt([a], level)

        sk_data = sk.data[0][self.ntt.starts[level][0]:]

        sa = self.ntt.mont_mult([a], [sk_data], level)
        self.ntt.intt_exit(sa, level)

        return sa

    def multiparty_decrypt_fusion(self, pcts: list,
                                  level: int = 0,
                                  include_special: bool = False,
                                  final_round: bool = True) -> np.ndarray:
        pt = [x.clone() for x in pcts[0]]
        for pct in pcts[1:]:
            pt = self.ntt.mont_add(pt, pct, level)

        self.ntt.reduce_2q(pt, level)

        if self.bias_guard:
            base_at = -self.ctx.num_special_primes - 1 if include_special else -1

            base = pt[0][base_at][None, :]
            scaler = pt[0][0][None, :]

            final_scalar = self.final_scalar[level]
            scaled = self.ntt.mont_sub([base], [scaler], -1)
            self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
            self.ntt.reduce_2q(scaled, -1)
            self.ntt.make_signed(scaled, -1)

        else:
            if level == self.num_levels - 1:
                base_at = -self.ctx.num_special_primes - 1 if include_special else -1

                base = pt[0][base_at][None, :]
                scaler = pt[0][0][None, :]

                final_scalar = self.final_scalar[level]
                scaled = self.ntt.mont_sub([base], [scaler], -1)
                self.ntt.mont_enter_scalar(scaled, [final_scalar], -1)
                self.ntt.reduce_2q(scaled, -1)
                self.ntt.make_signed(scaled, -1)

            else:  # level > self.num_levels - 1
                base = pt[0][1:]
                scaler = pt[0][0][None, :]

                final_scalar = self.new_final_scalar[level]

                scaled = base - scaler
                self.ntt.make_unsigned([scaled], lvl=level + 1)
                self.ntt.mont_enter_scalar([scaled], [final_scalar], lvl=level + 1)
                self.ntt.reduce_2q([scaled], lvl=level + 1)

                scaled = scaled[-2:]  # Two channels for decryption

                q1 = self.ctx.q[self.ntt.num_ordinary_primes - 2]
                q0 = self.ctx.q[self.ntt.num_ordinary_primes - 1]

                q1_inv_mod_q0_mont = self.new_final_scalar[self.ntt.num_ordinary_primes - 2]

                quotient = scaled[1:] - scaled[0]
                self.ntt.mont_enter_scalar([quotient], [q1_inv_mod_q0_mont], self.num_levels)
                self.ntt.reduce_2q([quotient], self.num_levels)

                M_half_div_q1 = (q0 - 1) // 2
                M_half_mod_q1 = (q1 - 1) // 2
                is_negative = torch.logical_or(
                    quotient[0] > M_half_div_q1,
                    torch.logical_and(
                        quotient[0] >= M_half_div_q1, scaled[0] > M_half_mod_q1
                    )
                )
                is_negative = is_negative * 1

                signed_large_part = quotient[0] - is_negative * q0
                scaled = [scaled[0].type(torch.float64) + float(q1) * signed_large_part.type(torch.float64)]

            if final_round:
                rounding_prime = self.ntt.qlists[0][-self.ctx.num_special_primes - 2]
                rounder = (scaler[0] > (rounding_prime // 2)) * 1
                scaled[0] += rounder

        # decode
        m = self.decode(m=scaled, level=level)

        return m

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. ROTATION
    #### -------------------------------------------------------------------------------------------

    def multiparty_create_key_switching_key(self, sk_src: DataStruct, sk_dst: DataStruct, crs=None) -> DataStruct:
        if sk_src.origin != types.origins["sk"] or sk_src.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin="not a secret key", to=types.origins["sk"])
        if (not sk_src.ntt_state) or (not sk_src.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_src.origin)
        if (not sk_dst.ntt_state) or (not sk_dst.montgomery_state):
            raise errors.NotMatchDataStructState(origin=sk_dst.origin)

        level = 0

        stops = self.ntt.stops[-1]
        Psk_src = [sk_src.data[di][:stops[di]].clone() for di in range(self.ntt.num_devices)]

        self.ntt.mont_enter_scalar(Psk_src, self.mont_PR, level)

        ksk = [[] for _ in range(self.ntt.p.num_partitions + 1)]
        for device_id in range(self.ntt.num_devices):
            for part_id, part in enumerate(self.ntt.p.p[level][device_id]):
                global_part_id = self.ntt.p.part_allocations[device_id][part_id]

                a = crs[global_part_id] if crs else None
                pk = self.multiparty_create_public_key(sk_dst, include_special=True, crs=a)
                key = tuple(part)
                astart = part[0]
                astop = part[-1] + 1
                shard = Psk_src[device_id][astart:astop]
                pk_data = pk.data[0][device_id][astart:astop]

                _2q = self.ntt.parts_pack[device_id][key]['_2q']
                update_part = ntt_cuda.mont_add([pk_data], [shard], _2q)[0]
                pk_data.copy_(update_part, non_blocking=True)

                # Name the pk.
                pk_name = f'key switch key part index {global_part_id}'
                pk = pk._replace(origin=pk_name)

                ksk[global_part_id] = pk

        return DataStruct(
            data=ksk,
            include_special=True,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["ksk"],
            level_calc=level,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )

    def multiparty_create_rotation_key(self, sk: DataStruct, delta: int, crs=None) -> DataStruct:
        sk_new_data = [s.clone() for s in sk.data]
        self.ntt.intt(sk_new_data)
        sk_new_data = [rotate(s, delta) for s in sk_new_data]
        self.ntt.ntt(sk_new_data)
        sk_rotated = DataStruct(
            data=sk_new_data,
            include_special=False,
            ntt_state=True,
            montgomery_state=True,
            origin=types.origins["sk"],
            level_calc=0,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )
        rotk = self.multiparty_create_key_switching_key(sk_rotated, sk, crs=crs)
        rotk = rotk._replace(origin=types.origins["rotk"] + f"{delta}")
        return rotk

    def multiparty_generate_rotation_key(self, rotks: list[DataStruct]) -> DataStruct:
        crotk = self.clone(rotks[0])
        for rotk in rotks[1:]:
            for ksk_idx in range(len(rotk.data)):
                update_parts = self.ntt.mont_add(crotk.data[ksk_idx].data[0], rotk.data[ksk_idx].data[0])
                crotk.data[ksk_idx].data[0][0].copy_(update_parts[0], non_blocking=True)
        return crotk

    def generate_rotation_crs(self, rotk: DataStruct):
        if types.origins["rotk"] not in rotk.origin and types.origins["ksk"] != rotk.origin:
            raise errors.NotMatchType(origin=rotk.origin, to=types.origins["ksk"])
        crss = []
        for ksk in rotk.data:
            crss.append(ksk.data[1])
        return crss

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. GALOIS
    #### -------------------------------------------------------------------------------------------

    def generate_galois_crs(self, galk: DataStruct):
        if galk.origin != types.origins["galk"]:
            raise errors.NotMatchType(origin=galk.origin, to=types.origins["galk"])
        crs_s = []
        for rotk in galk.data:
            crss = [ksk.data[1] for ksk in rotk.data]
            crs_s.append(crss)
        return crs_s

    def multiparty_create_galois_key(self, sk: DataStruct, crs: list) -> DataStruct:
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        galois_key_parts = [
            self.multiparty_create_rotation_key(sk, self.galois_deltas[idx], crs=crs[idx])
            for idx in range(len(self.galois_deltas))
        ]

        galois_key = DataStruct(
            data=galois_key_parts,
            include_special=True,
            montgomery_state=True,
            ntt_state=True,
            origin=types.origins["galk"],
            level_calc=0,
            level_available=self.num_levels,
            hash=self.strcp(self.hash),
            version=self.version
        )
        return galois_key

    def multiparty_generate_galois_key(self, galks: list[DataStruct]) -> DataStruct:
        cgalk = self.clone(galks[0])
        for galk in galks[1:]:  # galk
            for rotk_idx in range(len(galk.data)):  # rotk
                for ksk_idx in range(len(galk.data[rotk_idx].data)):  # ksk
                    update_parts = self.ntt.mont_add(
                        cgalk.data[rotk_idx].data[ksk_idx].data[0],
                        galk.data[rotk_idx].data[ksk_idx].data[0]
                    )
                    cgalk.data[rotk_idx].data[ksk_idx].data[0][0].copy_(update_parts[0], non_blocking=True)
        return cgalk

    #### -------------------------------------------------------------------------------------------
    #### Multiparty. Evaluation Key
    #### -------------------------------------------------------------------------------------------

    def multiparty_sum_evk_share(self, evks_share: list[DataStruct]):
        evk_sum = self.clone(evks_share[0])
        for evk_share in evks_share[1:]:
            for ksk_idx in range(len(evk_sum.data)):
                update_parts = self.ntt.mont_add(evk_sum.data[ksk_idx].data[0], evk_share.data[ksk_idx].data[0])
                for dev_id in range(len(update_parts)):
                    evk_sum.data[ksk_idx].data[0][dev_id].copy_(update_parts[dev_id], non_blocking=True)

        return evk_sum

    def multiparty_mult_evk_share_sum(self, evk_sum: DataStruct, sk: DataStruct):
        if sk.origin != types.origins["sk"]:
            raise errors.NotMatchType(origin=sk.origin, to=types.origins["sk"])
        evk_sum_mult = self.clone(evk_sum)

        for ksk_idx in range(len(evk_sum.data)):
            update_part_b = self.ntt.mont_mult(evk_sum_mult.data[ksk_idx].data[0], sk.data)
            update_part_a = self.ntt.mont_mult(evk_sum_mult.data[ksk_idx].data[1], sk.data)
            for dev_id in range(len(update_part_b)):
                evk_sum_mult.data[ksk_idx].data[0][dev_id].copy_(update_part_b[dev_id], non_blocking=True)
                evk_sum_mult.data[ksk_idx].data[1][dev_id].copy_(update_part_a[dev_id], non_blocking=True)

        return evk_sum_mult

    def multiparty_sum_evk_share_mult(self, evk_sum_mult: list[DataStruct]) -> DataStruct:
        cevk = self.clone(evk_sum_mult[0])
        for evk in evk_sum_mult[1:]:
            for ksk_idx in range(len(cevk.data)):
                update_part_b = self.ntt.mont_add(cevk.data[ksk_idx].data[0], evk.data[ksk_idx].data[0])
                update_part_a = self.ntt.mont_add(cevk.data[ksk_idx].data[1], evk.data[ksk_idx].data[1])
                for dev_id in range(len(update_part_b)):
                    cevk.data[ksk_idx].data[0][dev_id].copy_(update_part_b[dev_id], non_blocking=True)
                    cevk.data[ksk_idx].data[1][dev_id].copy_(update_part_a[dev_id], non_blocking=True)
        return cevk

    #### -------------------------------------------------------------------------------------------
    ####  Statistics
    #### -------------------------------------------------------------------------------------------

    def mean(self, ct, gk, alpha=1):
        """
            Divide by num_slots.
            The cipher text is refreshed here, and hence
            doesn't need to be refreshed at roti=0 in the loop.
        @param ct:
        @param gk:
        @param alpha:
        @return:
        """

        new_ct = self.mult(1 / self.num_slots / alpha, ct)

        for roti in range(self.ctx.logN - 1):
            rotk = gk.data[roti]
            rot_ct = self.rotate_single(new_ct, rotk)
            new_ct = self.add(rot_ct, new_ct)
        return new_ct

    def cov(self, ct_a: DataStruct, ct_b: DataStruct,
            evk: DataStruct, gk: DataStruct) -> DataStruct:
        cta_mean = self.mean(ct_a, gk)
        ctb_mean = self.mean(ct_b, gk)

        cta_dev = self.sub(ct_a, cta_mean)
        ctb_dev = self.sub(ct_b, ctb_mean)

        ct_cov = self.mult(self.mult(cta_dev, ctb_dev, evk), 1 / (self.num_slots - 1))

        return ct_cov

    def sqrt(self, ct: DataStruct, evk: DataStruct, e=0.0001, alpha=0.0001) -> DataStruct:
        a = self.clone(ct)
        b = self.clone(ct)

        while e <= 1 - alpha:
            k = float(np.roots([1 - e ** 3, -6 + 6 * e ** 2, 9 - 9 * e])[1])
            t = self.mult_scalar(a, k, evk)
            b0 = self.sub_scalar(t, 3)
            b1 = self.mult_scalar(b, (k ** 0.5) / 2, evk)
            b = self.cc_mult(b0, b1, evk)

            a0 = self.mult_scalar(a, (k ** 3) / 4)
            t = self.sub_scalar(a, 3 / k)
            a1 = self.square(t, evk)
            a = self.cc_mult(a0, a1, evk)
            e = k * (3 - k) ** 2 / 4

        return b

    def var(self, ct: DataStruct, evk: DataStruct, gk: DataStruct, relin=False) -> DataStruct:
        ct_mean = self.mean(ct=ct, gk=gk)
        dev = self.sub(ct, ct_mean)
        dev = self.square(ct=dev, evk=evk, relin=relin)
        if not relin:
            dev = self.relinearize(ct_triplet=dev, evk=evk)
        ct_var = self.mean(ct=dev, gk=gk)
        return ct_var

    def std(self, ct: DataStruct, evk: DataStruct, gk: DataStruct, relin=False) -> DataStruct:
        ct_var = self.var(ct=ct, evk=evk, gk=gk, relin=relin)
        ct_std = self.sqrt(ct=ct_var, evk=evk)
        return ct_std
