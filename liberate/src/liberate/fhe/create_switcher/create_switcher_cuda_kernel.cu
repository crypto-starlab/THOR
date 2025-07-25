#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

#include "create_switcher_cuda.h"

#include <iostream>

using namespace std;

#define BLOCK_SIZE 256

// inclusion of ntt kernels must come after the definition of BLOCK_SIZE.
#include "ntt_cuda_kernels.h"


//////////////////////////////////////////////////////////////////////////
// CUDA kernels.


template<typename scalar_t>
__global__ void initialize_state(
    scalar_t* a,
    scalar_t* state,
    int N
    ) {

    // Calculate my position in state.
    // i is in the direction of RNS channels, and
    // j is in the direction of the polynomial coefficients.
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    // Assign.
    // Repeat the channel0 elements in every channel of the state.
    state[i*N + j] = a[j];
}




template<typename scalar_t>
__global__ void update_state_row(
    scalar_t* a,
    scalar_t* state,
    int N, int i,
    scalar_t* Y_scalar,
    scalar_t* ql, scalar_t* qh,
    scalar_t* kl, scalar_t* kh
    ) {
    const int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    const int ind = (i+1)*N + j;

    // auto Y will promote the type of int8_t to int.
    // Explicitly specify the type of Y.
    scalar_t Y = a[ind] - state[(i+1)*N+j];

    state[ind] = mont_mult_scalar(
        Y,
        Y_scalar[0],
        ql[0], qh[0], kl[0], kh[0]
    );
}



template<typename scalar_t>
__global__ void update_state_rest(
    scalar_t* state,
    int i0, int N,
    scalar_t* L_scalar,
    scalar_t* ql, scalar_t* qh,
    scalar_t* kl, scalar_t* kh
) {
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    scalar_t Y = state[(i0+1)*N+j];
    state[(i0+i+2)*N + j] += mont_mult_scalar(
        Y, L_scalar[i],
        ql[i], qh[i], kl[i], kh[i]
    );

}



template<typename scalar_t>
__global__ void initialize_extended(
    scalar_t* state,
    scalar_t* extended,
    int N,
    scalar_t* Rs,
    scalar_t* ql,
    scalar_t* qh,
    scalar_t* kl,
    scalar_t* kh
    )
{
    // Calculate my position in state.
    // i is in the direction of RNS channels, and
    // j is in the direction of the polynomial coefficients.
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    // Assign.
    // Repeat the first row in the extended.
    extended[i*N + j] = mont_mult_scalar(
        state[j], Rs[i], ql[i], qh[i], kl[i], kh[i]
    );
}

template<typename scalar_t>
__global__ void update_extended(
    scalar_t* state,
    scalar_t* extended,
    scalar_t* L_enter,
    int N,
    int alpha_i,
    scalar_t* _2q,
    scalar_t* ql,
    scalar_t* qh,
    scalar_t* kl,
    scalar_t* kh
){
    // Calculate my position in state.
    // i is in the direction of RNS channels, and
    // j is in the direction of the polynomial coefficients.
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    const int ind = i*N + j;

    auto new_value = extended[ind] + mont_mult_scalar(
        state[(alpha_i+1)*N + j], L_enter[i], ql[i], qh[i], kl[i], kh[i]
    );

    // Reduce to 0~2q range.
    auto my_2q = _2q[i];
    extended[ind] = (new_value < my_2q)? new_value : new_value - my_2q;

}

template<typename scalar_t>
__global__ void mont_mult_add_update(
    scalar_t* extended,
    scalar_t* ksk0,
    scalar_t* ksk1,
    scalar_t* switcher0,
    scalar_t* switcher1,
    int N,
    scalar_t* _2q,
    scalar_t* ql,
    scalar_t* qh,
    scalar_t* kl,
    scalar_t* kh
){
    // Calculate my position in state.
    // i is in the direction of RNS channels, and
    // j is in the direction of the polynomial coefficients.
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    const int ind = i*N + j;

    scalar_t d0 = mont_mult_scalar(
        extended[ind], ksk0[ind], ql[i], qh[i], kl[i], kh[i]
    );

    scalar_t d1 = mont_mult_scalar(
        extended[ind], ksk1[ind], ql[i], qh[i], kl[i], kh[i]
    );

    // Multiple streams may be attempting to update the switch.
    // Use atomic operation.
    using atomic_t = atomicCASType<scalar_t>;

    atomic_t* addr =
        reinterpret_cast<atomic_t*>(&switcher0[ind]);

    atomic_t old = *addr, assumed;
    scalar_t new_val;

    // We need _2q for reduction.
    scalar_t my_2q = _2q[i];


    // Update s0.
    do {
        assumed = old;

        // Calculate the update based on the assumed value.
        new_val = static_cast<scalar_t>(assumed) + d0;
        new_val = (new_val < my_2q)? new_val : new_val - my_2q;

        // Compare and swap.
        old = atomicCAS(addr, assumed, static_cast<atomic_t>(new_val));
    } while (assumed != old);

    // Update s1.
    addr = reinterpret_cast<atomic_t*>(&switcher1[ind]);
    old = *addr;


    do {
        assumed = old;

        // Calculate the update based on the assumed value.
        new_val = static_cast<scalar_t>(assumed) + d1;
        new_val = (new_val < my_2q)? new_val : new_val - my_2q;

        // Compare and swap.
        old = atomicCAS(addr, assumed, static_cast<atomic_t>(new_val));
    } while (assumed != old);

}


template<typename scalar_t>
__global__ void divide_constrict(
    scalar_t* switcher,
    scalar_t* PiRi,
    int N,
    int divisor_loc,
    scalar_t* _2q_ptr,
    scalar_t* ql_ptr,
    scalar_t* qh_ptr,
    scalar_t* kl_ptr,
    scalar_t* kh_ptr
){
    const int i = blockIdx.x;
    const int j = blockIdx.y * BLOCK_SIZE + threadIdx.x;

    // Parameters
    scalar_t PiR = PiRi[i];
    scalar_t _2q = _2q_ptr[i];
    scalar_t ql = ql_ptr[i];
    scalar_t qh = qh_ptr[i];
    scalar_t kl = kl_ptr[i];
    scalar_t kh = kh_ptr[i];

    // ind.
    const int ind = i*N + j;

    // Get the divisor and the dividend.
    scalar_t divisor = switcher[divisor_loc*N + j];
    scalar_t dividend = switcher[ind];

    // Mont enter both the dividend and the divisor.
    // Since we are multiplying Pinv*R, we will end up
    // with the divisor*Pinv mod q, the usual reduction,
    // not int the montgomery form.
    divisor = mont_mult_scalar(divisor, PiR, ql, qh, kl, kh);
    dividend = mont_mult_scalar(dividend, PiR, ql, qh, kl, kh);

    // Update the dividend by subtracting the divisor.
    dividend = dividend - divisor + _2q;

    // Reduce to 0~2q.
    dividend = (dividend < _2q)? dividend : dividend - _2q;

    // Reduce to 0~q.
    const scalar_t q = _2q >> 1;
    dividend = (dividend < q)? dividend : dividend - q;

    // Store back to switcher.
    switcher[ind] = dividend;
}



/*-----------------------------------------------------------------------*/






//////////////////////////////////////////////////////////////////////////
// Extend, multiply and add, the state.

template<typename scalar_t>
// __attribute__((always_inline)) inline void extend_and_on(
void extend_and_on(
    int did, int pid,
    int alpha, int N,
    scalar_t* state,
    scalar_t* extended,
    std::vector<std::vector<std::vector<torch::Tensor>>>& ksk,
    std::vector<std::vector<torch::Tensor>>& switcher,
    size_t params, int cuda_device, int target_device_id, int execution_order, cudaStream_t stream
){

    // Convert the params.
    create_switcher_params& p = *reinterpret_cast<create_switcher_params*>(params);

    // Derive the number of channels.
    auto C = p.Rs_prepack_plan[target_device_id].size(0);

    // Set the device.
    cudaSetDevice(cuda_device);

    ////////////////////////////////////////////////////////////////
    // Extend.

    // Initialize.
    auto Rs = p.Rs_prepack_plan[target_device_id].data_ptr<scalar_t>();
    auto mont_pack = p.mont_prepack_plan[target_device_id];
    auto ql = mont_pack[0].data_ptr<scalar_t>();
    auto qh = mont_pack[1].data_ptr<scalar_t>();
    auto kl = mont_pack[2].data_ptr<scalar_t>();
    auto kh = mont_pack[3].data_ptr<scalar_t>();

    int dim_block = BLOCK_SIZE;
    dim3 dim_grid(C, N / BLOCK_SIZE);
    initialize_extended<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
        state, extended, N, Rs, ql, qh, kl, kh
    );

    // L_enter.
    // Note that L_enter may be empty.
    auto L_enter = p.L_enter_plan[did][pid][execution_order];

    // _2q.
    // We need for mont add.
    auto _2q = p._2q_plan[target_device_id].data_ptr<scalar_t>();

    // Iterate over alpha.
    for(int i=0; i < alpha-1; i++){
        update_extended<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
            state, extended, L_enter[i].data_ptr<scalar_t>(),
            N, i, _2q, ql, qh, kl, kh
        );
    }

    // ntt the extended.
    auto ntt_prepack = p.ntt_prepack_plan[target_device_id];

    const auto even_acc = ntt_prepack[0].packed_accessor32<int, 2>();
    const auto odd_acc = ntt_prepack[1].packed_accessor32<int, 2>();
    const auto psi_acc = ntt_prepack[2].packed_accessor32<scalar_t, 3>();

    // Get _2q, ql, qh, kl, kh for the full range of RNS.
    _2q = ntt_prepack[3].data_ptr<scalar_t>();
    ql = ntt_prepack[4].data_ptr<scalar_t>();
    qh = ntt_prepack[5].data_ptr<scalar_t>();
    kl = ntt_prepack[6].data_ptr<scalar_t>();
    kh = ntt_prepack[7].data_ptr<scalar_t>();

    // even.size(0)
    const int logN = ntt_prepack[0].size(0);

    // Watch out!!!
    // We're using butterfly for ntt.
    // That is, we process even and odd indexed terms both at the same time.
    // Hence, the total length of the iteration is N/2.
    // Reset the block and grid sizes.
    dim_block = BLOCK_SIZE;
    dim_grid = dim3(C, N / BLOCK_SIZE / 2);

    for(int i=0; i<logN; ++i){
        ntt_cuda_kernel<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
        extended, N,
        even_acc, odd_acc, psi_acc,
        _2q, ql, qh, kl, kh, i);
    }

    // Multiply with the ksk and add to the switcher.
    auto ksk_loc = p.ksk_loc_plan[did][pid];
    scalar_t* ksk0 = ksk[ksk_loc][0][target_device_id].data_ptr<scalar_t>();
    scalar_t* ksk1 = ksk[ksk_loc][1][target_device_id].data_ptr<scalar_t>();

    // We use partial ksk.
    int ksk_start = p.ksk_starts_plan[target_device_id];
    ksk0 += ksk_start * N;
    ksk1 += ksk_start * N;

    scalar_t* switcher0 = switcher[0][target_device_id].data_ptr<scalar_t>();
    scalar_t* switcher1 = switcher[1][target_device_id].data_ptr<scalar_t>();

    // Reset the dim_grid.
    dim_block = BLOCK_SIZE;
    dim_grid = dim3(C, N / BLOCK_SIZE);
    mont_mult_add_update<scalar_t><<<dim_grid, dim_block, 0, stream>>>(
        extended, ksk0, ksk1, switcher0, switcher1, N, _2q,
        ql, qh, kl, kh
    );

  }




//////////////////////////////////////////////////////////////////////////
// The main part processor.

template<typename scalar_t>
void process_part_typed(
    size_t did, size_t pid,
    std::vector<torch::Tensor>& a,
    std::vector<std::vector<std::vector<torch::Tensor>>>& ksk,
    std::vector<std::vector<torch::Tensor>>& switcher,
    size_t params, size_t streams
) {
    // Unpack parameters.
    create_switcher_params& p = *reinterpret_cast<create_switcher_params*>(params);

    // Unpack streams
    cs_streams_t& s = *reinterpret_cast<cs_streams_t*>(streams);

    // Retrieve the basic information.
    const auto alpha = p.part_plan[did][pid][0];
    const auto channel0 = p.part_plan[did][pid][1];

    // Devices.
    // Watch out!!!! device_ids and cuda devices are different!!!
    // device_id=0 may indicate cuda_device=1.
    std::vector<int>& cuda_devices = p.run_device_plan[did];
    std::vector<int>& device_ids = p.device_id_plan[did];

    // Tensor information.
    // Downcast size_t to int.
    // int indexing is faster in cuda.
    int N = a[did].size(1);

    // Note that device_id and cuda devices are different!!!
    // For example, if device_ids = [0, 1, 2] were mapped to cuda devices [3, 0, 1],
    // device_id of value 0 corresponds to cuda device 3.
    // In fact device_ids denote the order of execution, such that
    // device_ids = [1, 0, 2] -> cuda devices [0, 3, 1] given the above example.

    // The correct source cuda device is always stored at cuda_devices[0].
    // That is the cuda device of device_ids[0].
    // The rest are neighbor devices.
    // Also, device_ids[0] is always did, since it is the source device_id.
    auto source_cuda_device = cuda_devices[0];

    cudaSetDevice(source_cuda_device);

    // We will repeatedly use the stream on the compute stream at did.
    auto main_stream = s[did][pid].compute[did];

    // We will create multiple copies of the state.
    // Precalculate the bytesize.
    size_t state_byte_size = sizeof(scalar_t) * alpha * N;

    // Pull my state share.
    scalar_t* full_state = p.states[did][did].data_ptr<scalar_t>();

    // My share of state
    scalar_t* state = full_state + channel0 * N;

    // My share of a.
    scalar_t* my_a = a[did].data_ptr<scalar_t>() + channel0 * N;

    // Generate state.
    // Partition the alpha X N size memory block
    // into (alpha X (N/BLOCK_SIZE)) X BLOCK_SIZE,
    // so that in a thread block, memory access is contiguous.

    const int dim_block = BLOCK_SIZE;
    dim3 dim_grid(alpha, N / BLOCK_SIZE);

    initialize_state<scalar_t><<<dim_grid, dim_block, 0, main_stream>>>(
        my_a,
        state,
        N
    );

    auto mont_packs = p.mont_plan[did][pid];

    for(int i=0; i< alpha-1; i++) {

        auto mont_pack = mont_packs[i];
        auto ql = mont_pack[0].data_ptr<scalar_t>();
        auto qh = mont_pack[1].data_ptr<scalar_t>();
        auto kl = mont_pack[2].data_ptr<scalar_t>();
        auto kh = mont_pack[3].data_ptr<scalar_t>();
        // Y_scalar is a number wrapped in a tensor.
        // Its size is 1.
        auto Y_scalar = p.Y_plan[did][pid][i].data_ptr<scalar_t>();

        const int dim_grid = N / BLOCK_SIZE;
        update_state_row<scalar_t><<<dim_grid, dim_block, 0, main_stream>>>
        (
            my_a,
            state,
            N, i,
            Y_scalar,
            ql, qh, kl, kh
        );

        if ((i+2) < alpha) {
            auto new_state_len = alpha - (i+2);
            auto mont_pack = p.state_mont_plan[did][pid][i];
            auto ql = mont_pack[0].data_ptr<scalar_t>();
            auto qh = mont_pack[1].data_ptr<scalar_t>();
            auto kl = mont_pack[2].data_ptr<scalar_t>();
            auto kh = mont_pack[3].data_ptr<scalar_t>();
            auto L_scalar = p.L_scalar_plan[did][pid][i].data_ptr<scalar_t>();
            dim3 dim_grid(new_state_len, dim_block);
            update_state_rest<scalar_t><<<dim_grid, dim_block, 0, main_stream>>>
            (
                state, i, N, L_scalar,
                ql, qh, kl, kh
            );
        }
    }

    // State generation is complete.
    // Record an event.
    cudaEvent_t d2h_start_event;
    cudaEventCreate(&d2h_start_event);
    cudaEventRecord(d2h_start_event, main_stream);

    ////////////////////////////////////////////////////////////////
    // Staging: Copy to CPU.

    // Device to host stream.
    auto d2h_stream = s[did][pid].d2h;

    // Wait for the state generation to complete.
    cudaStreamWaitEvent(d2h_stream, d2h_start_event);
    cudaEventDestroy(d2h_start_event);

    // Book a transfer to CPU.
    // CPU_state_plan contains pinned memory_buffers.

    auto cpu_buffer = p.CPU_state_plan[did][pid].data_ptr<scalar_t>();
    cudaMemcpyAsync(
        reinterpret_cast<void*>(cpu_buffer),
        reinterpret_cast<const void*>(state),
        state_byte_size,
        cudaMemcpyDeviceToHost,
        d2h_stream
    );

    cudaEvent_t d2h_end_event;
    cudaEventCreate(&d2h_end_event);
    cudaEventRecord(d2h_end_event, d2h_stream);

    ////////////////////////////////////////////////////////////////
    // Continue on with the main state.

    // My extend.
    int ksk_loc = p.ksk_loc_plan[did][pid];
    scalar_t* extended = p.extends[ksk_loc][did].data_ptr<scalar_t>();

    extend_and_on(
        did, pid, alpha, N, state, extended, ksk, switcher,
        params, source_cuda_device, did, 0, main_stream
    );



    ////////////////////////////////////////////////////////////////
    // Continue on, at neighbor devices.

    // Start the loop from 1. 0 is the source device.
    for(int i=1; i<cuda_devices.size(); i++) {
        auto cudev = cuda_devices[i];
        auto devid = device_ids[i];

        auto target_stream = s[did][pid].compute[devid];

        // Set device.
        cudaSetDevice(cudev);

        // My target state.
        scalar_t* target_full_state = p.states[did][devid].data_ptr<scalar_t>();
        scalar_t* target_state = target_full_state + channel0 * N;

        // Wait for the d2h transfer completion, and then start fetching.
        cudaStreamWaitEvent(target_stream, d2h_end_event);

        cudaMemcpyAsync(
            reinterpret_cast<void*>(target_state),
            reinterpret_cast<const void*>(cpu_buffer),
            state_byte_size,
            cudaMemcpyHostToDevice,
            target_stream
        );

        ////////////////////////////////////////////////////////////////
        // Do something with the transferred state.

        scalar_t* target_extended = p.extends[ksk_loc][devid].data_ptr<scalar_t>();

        extend_and_on(
            did, pid, alpha, N, target_state, target_extended, ksk, switcher,
            params, cudev, devid, i, target_stream
        );


    }

    ////////////////////////////////////////////////////////////////
    // Destroy the d2h_end_event.
    cudaEventDestroy(d2h_end_event);

}






//////////////////////////////////////////////////////////////////////////
// The division and finalization function.


template<typename scalar_t>
void divide_and_finalize(
    int num_special_primes,
    std::vector<std::vector<torch::Tensor>>& switcher,
    size_t params
) {

    create_switcher_params& p = *reinterpret_cast<create_switcher_params*>(params);

    // We need to set the device.
    // p.run_devices_plan[0] is always in the ascending order
    // of cuda devices.
    auto cuda_devices = p.run_device_plan[0];

    // N and half_N.
    const int N = p.states[0][0].size(1);
    const int half_N = N >> 1;

    // intt the summed switcher.
    // We do this in the default stream.
    // intt the switcher.
    for(size_t did=0; did<p.part_plan.size(); did++){

        // Don't forget to set device!!!!
        cudaSetDevice(cuda_devices[did]);

        auto intt_prepack = p.intt_prepack_plan[did];

        const auto even_acc = intt_prepack[0].packed_accessor32<int, 2>();
        const auto odd_acc = intt_prepack[1].packed_accessor32<int, 2>();
        const auto psi_acc = intt_prepack[2].packed_accessor32<scalar_t, 3>();

        // Get Ninv, _2q, ql, qh, kl, kh for the full range of RNS.
        scalar_t* Ninv = intt_prepack[3].data_ptr<scalar_t>();
        scalar_t* _2q = intt_prepack[4].data_ptr<scalar_t>();
        scalar_t* ql = intt_prepack[5].data_ptr<scalar_t>();
        scalar_t* qh = intt_prepack[6].data_ptr<scalar_t>();
        scalar_t* kl = intt_prepack[7].data_ptr<scalar_t>();
        scalar_t* kh = intt_prepack[8].data_ptr<scalar_t>();

        // even.size(0)
        const int logN = intt_prepack[0].size(0);

        // C and N.
        int C = p.Rs_prepack_plan[did].size(0);

        // Grid.
        int dim_block = BLOCK_SIZE;
        dim3 dim_grid_ntt (C, half_N / BLOCK_SIZE);
        dim3 dim_grid_enter (C, N / BLOCK_SIZE);

        for(int texti=0; texti<2; texti++)
        {
            scalar_t* my_switcher = switcher[texti][did].data_ptr<scalar_t>();

            for(int i=0; i<logN; ++i)
            {
                intt_cuda_kernel<scalar_t><<<dim_grid_ntt, dim_block>>>
                (
                    my_switcher,
                    N,
                    even_acc, odd_acc, psi_acc,
                    _2q, ql, qh, kl, kh, i
                );
            }

            // Normalize.
            normalize_exit_reduce_cuda_kernel<scalar_t><<<dim_grid_enter, dim_block>>>
            (
                my_switcher, N, Ninv, _2q, ql, qh, kl, kh
            );

            // Divide by P.
            for(int Pind=0; Pind<num_special_primes; Pind++) {

                // PiRi.
                scalar_t* PiRi = p.PiRi_plan[Pind][did].data_ptr<scalar_t>();

                // C changes dynamically.
                // The last channel in the constricted tensor
                // contains the divisor.
                // Hence -1 to compensate for the divisor.
                // We do not operate on the channel in the kernel.
                int constricted_C = C - Pind - 1;

                int dim_block = BLOCK_SIZE;
                dim3 dim_grid_divcon (constricted_C, N / BLOCK_SIZE);

                divide_constrict<scalar_t><<<dim_grid_divcon, dim_block>>>(
                    my_switcher, PiRi, N, constricted_C,
                    _2q, ql, qh, kl, kh
                );
            }
        }
    }
}








//////////////////////////////////////////////////////////////////////////
// The wrapping connector.

// The main function of create_switcher on cuda.
void create_switcher_cuda_main(
    std::vector<torch::Tensor>& a,
    std::vector<std::vector<std::vector<torch::Tensor>>>& ksk,
    std::vector<std::vector<torch::Tensor>>& switcher,
    size_t params, size_t streams
) {
    // Unpack parameters.
    create_switcher_params& p = *reinterpret_cast<create_switcher_params*>(params);
    // cs_streams_t& s = *reinterpret_cast<cs_streams_t*>(streams);

    // Iterate over source devices.
    for(size_t did=0; did<p.part_plan.size(); did++){
        // Iterate over partitions.
        for(size_t pid=0; pid<p.part_plan[did].size(); pid++) {
            // Dispatch.
            AT_DISPATCH_INTEGER_TYPES(a[did].scalar_type(), "typed_process_part", ([&] {
                process_part_typed<scalar_t>(
                    did, pid, a, ksk, switcher, params, streams
                );
            }));
        }
    }

    // Derive the num_special_primes.
    int num_special_primes = p.extends[0][0].size(0) - p.states[0][0].size(0);

    // Finalize.
    AT_DISPATCH_INTEGER_TYPES(a[0].scalar_type(), "typed_divide_and_finalize", ([&] {
        divide_and_finalize<scalar_t>(num_special_primes, switcher, params);
    }));

    // We don't need to sync devices as we finalized our calculation
    // on the default stream.

// End of create_switcher_cuda_main.
}
