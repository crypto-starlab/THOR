import os
import re
from types import NoneType
from typing import Dict, Union

import torch


def allocated_gpu_memory(
        devices: Union[list[int], NoneType] = None,
        MB: bool = True, empty_cache: bool = True) -> Dict[str, float]:
    """
        This function returns the amount of memory in bytes currently allocated by the active PyTorch tensors on the specified CUDA device
        It provides the amount of GPU memory actively used by tensors created in your PyTorch program.
        It focuses on the memory usage of your current PyTorch program (specifically the tensors).
        - Devices
            - None : All devices
            - [int] : Numbers of the device you want to use (e.g. [0, 1], [1, 3])
    @param devices: the number list of the device you want to use.
    @param MB:
    @param empty_cache:
    @return:
    """
    if devices is None:
        devices = [f"cuda:{x}" for x in range(torch.cuda.device_count())]

    gpu_memories = dict()
    if torch.cuda.is_available():
        if empty_cache:
            torch.cuda.empty_cache()

        for device in devices:
            mem = torch.cuda.memory_allocated(device)
            gpu_memories[device] = mem / (1024 ** 2) if MB else mem
            print(f"{device}\t\t: {mem / (1024 ** 2):>8.2f} MB")
        sum_memories = sum(gpu_memories.values())
        gpu_memories["all"] = sum_memories
        print(f"all devices\t: {sum_memories:>8.2f} MB")
    else:
        raise "Gpu devices is not available"

    return gpu_memories


def list_gpu_processes(devices: Union[list[int], NoneType] = None):
    """
        This function provides information about working processes currently using the CUDA device.
         - Devices
            - None : All devices
            - [int] : Numbers of the device you want to use (e.g. [0, 1], [1, 3])
    @param devices:  the number list of the device you want to use.
    @return:
    """
    if devices is None:
        devices = [f"cuda:{x}" for x in range(torch.cuda.device_count())]

    if torch.cuda.is_available():
        pid = os.getpid()
        print(f"pid\t\t: {pid:>8d}")
        c = re.compile("\d+[.]\d+")
        result = {}
        for device in devices:
            gpu_memories_pid = [x for x in torch.cuda.memory.list_gpu_processes(device).split("\n") if str(pid) in x]
            for gpu_memory in gpu_memories_pid:
                gpu_mem_pid = float(c.findall(gpu_memory)[0])
                print(f"{device}\t\t: {gpu_mem_pid:>8.2f} MB")
                result[device] = gpu_mem_pid
        sum_memories = sum(result.values())
        result["all"] = sum_memories
        print(f"all devices\t: {sum_memories:>8.2f} MB")
    else:
        raise "Gpu devices is not available"

    return result
