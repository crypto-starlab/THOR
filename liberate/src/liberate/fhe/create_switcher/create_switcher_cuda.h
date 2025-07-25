#ifndef CREATE_SWITCHER_CUDA_H
#define CREATE_SWITCHER_CUDA_H

#include <torch/extension.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include <type_traits>

// We're dealing with a huge list of parameters.
// Aggregate them in a struct.
struct create_switcher_params {
    std::vector<std::vector<std::vector<int>>> part_plan;
	std::vector<std::vector<std::vector<std::vector<torch::Tensor>>>> mont_plan;
	std::vector<std::vector<std::vector<torch::Tensor>>> Y_plan;
	std::vector<std::vector<std::vector<std::vector<torch::Tensor>>>> state_mont_plan;
	std::vector<std::vector<std::vector<torch::Tensor>>> L_scalar_plan;
	std::vector<std::vector<int>> run_device_plan;
    std::vector<std::vector<int>> device_id_plan;
	std::vector<std::vector<torch::Tensor>> CPU_state_plan;
	std::vector<torch::Tensor> Rs_prepack_plan;
	std::vector<std::vector<torch::Tensor>> mont_prepack_plan;
	std::vector<std::vector<std::vector<std::vector<torch::Tensor>>>> L_enter_plan;
	std::vector<torch::Tensor> _2q_plan;
	std::vector<std::vector<torch::Tensor>> ntt_prepack_plan;
	std::vector<std::vector<torch::Tensor>> intt_prepack_plan;
	std::vector<std::vector<int>> ksk_loc_plan;
	std::vector<int> ksk_starts_plan;
	std::vector<std::vector<torch::Tensor>> PiRi_plan;
    std::vector<std::vector<torch::Tensor>> states;
    std::vector<std::vector<torch::Tensor>> extends;
};

// Pre-allocate cuda streams, so that we won't have to create and destroy every time.
// The struct is per a partition.
// Eventually, we will end up with a vector of the struct, one for each partition.
struct create_switcher_streams {
    std::vector<cudaStream_t> compute;
    cudaStream_t d2h;
};

// We will use an array of arrays for the stream struct.
using cs_streams_t = std::vector<std::vector<create_switcher_streams>>;


// We are using atomicCAS.
// Unfortunately, the CUDA API doesn't permit usage of 8bit integers for the function.
// However, AT_DISPATCH_INTEGRAL_TYPES attempts to instantiate all the available
// integer formats.
// Here, use a modified version of the AT_DISPATCH_INTEGRAL_TYPES,
// AT_DISPATCH_INTEGER_TYPES that cover 16, 32, and 64 bit integers.

#define AT_DISPATCH_CASE_INTEGER_TYPES(...)          \
  AT_DISPATCH_CASE(at::ScalarType::Int, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::Long, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Short, __VA_ARGS__)

#define AT_DISPATCH_INTEGER_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, AT_DISPATCH_CASE_INTEGER_TYPES(__VA_ARGS__))

// atomicCAS permits 3 types of integer types as an input and an output.
// We need a way to convert a type, according to its byte length, to a proper type.
template<typename T>
using atomicCASType = typename std::conditional<
    sizeof(T)==2,
    unsigned short int,
    typename std::conditional<
        sizeof(T)==4,
        unsigned int,
        typename std::conditional<
            sizeof(T)==8,
            unsigned long long int,
            void
        >::type
    >::type
>::type;

#endif /*CREATE_SWITCHER_CUDA_H */
