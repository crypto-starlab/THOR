#include <torch/extension.h>
#include <vector>

#include "create_switcher_cuda.h"


#include <iostream>
using namespace std;


/****************************************************************************

Utility functions: Preallocate plan parameters and delete.

*****************************************************************************/


// We're dealing with a huge list of parameters.
// Aggregate them in a struct.
size_t create_cs_params_ptr(
    std::vector<std::vector<std::vector<int>>> part_plan,
	std::vector<std::vector<std::vector<std::vector<torch::Tensor>>>> mont_plan,
	std::vector<std::vector<std::vector<torch::Tensor>>> Y_plan,
	std::vector<std::vector<std::vector<std::vector<torch::Tensor>>>> state_mont_plan,
	std::vector<std::vector<std::vector<torch::Tensor>>> L_scalar_plan,
	std::vector<std::vector<int>> run_device_plan,
    std::vector<std::vector<int>> device_id_plan,
	std::vector<std::vector<torch::Tensor>> CPU_state_plan,
	std::vector<torch::Tensor> Rs_prepack_plan,
	std::vector<std::vector<torch::Tensor>> mont_prepack_plan,
	std::vector<std::vector<std::vector<std::vector<torch::Tensor>>>> L_enter_plan,
	std::vector<torch::Tensor> _2q_plan,
	std::vector<std::vector<torch::Tensor>> ntt_prepack_plan,
	std::vector<std::vector<torch::Tensor>> intt_prepack_plan,
	std::vector<std::vector<int>> ksk_loc_plan,
	std::vector<int> ksk_starts_plan,
	std::vector<std::vector<torch::Tensor>> PiRi_plan,
    std::vector<std::vector<torch::Tensor>> states,
    std::vector<std::vector<torch::Tensor>> extends
) {
    create_switcher_params *params = new create_switcher_params {
        part_plan, mont_plan, Y_plan,
        state_mont_plan, L_scalar_plan,
        run_device_plan, device_id_plan, CPU_state_plan,
        Rs_prepack_plan, mont_prepack_plan, L_enter_plan,
        _2q_plan, ntt_prepack_plan, intt_prepack_plan,
        ksk_loc_plan, ksk_starts_plan, PiRi_plan, states, extends
    };

    size_t params_ptr = reinterpret_cast<size_t>(params);

    return params_ptr;
}

// Of course, we should be able to free the allocated memory.
void delete_cs_params_ptr(size_t params) {
    create_switcher_params* params_ptr = reinterpret_cast<create_switcher_params*>(params);
    delete params_ptr;
}

/****************************************************************************

Utility functions: Preallocate streams and destroy.

*****************************************************************************/


// We pre-allocate streams.
size_t preallocate_streams(std::vector<int> num_parts, std::vector<int> cuda_devices) {

    // The composition.
    // [device_id][part_id]
    cs_streams_t* device_part_streams =
        new cs_streams_t;

    // Make an alias for easier access.
    cs_streams_t& dps = *device_part_streams;

    for(size_t did=0; did<num_parts.size(); did++) {

        int cdev = cuda_devices[did];
        std::vector<create_switcher_streams> part_streams;

        for(int pid=0; pid<num_parts[did]; pid++) {

            // d2h copy starts at did.
            cudaSetDevice(cdev);
            create_switcher_streams css;
            cudaStreamCreate(&(css.d2h));

            // Assign a stream per a device in the compute streams vector.
            // Compute streams are index by the device_id.
            // sub_did covers all available devices.
            for(size_t sub_did=0; sub_did<num_parts.size(); sub_did++) {

                int sub_cdev = cuda_devices[sub_did];

                cudaSetDevice(sub_cdev);
                cudaStream_t new_stream;
                cudaStreamCreate(&new_stream);
                css.compute.push_back(new_stream);
            }
            part_streams.push_back(css);
        }
        dps.push_back(part_streams);
    }

    // Convert to id. That is size_t.
    size_t id = reinterpret_cast<size_t>(device_part_streams);

    return id;
}

// Of course we need a way to destroy the streams and free the allocated memory
// for the vector.
void destroy_streams(size_t device_part_streams_id) {
    cs_streams_t* sptr = reinterpret_cast<cs_streams_t*>(device_part_streams_id);
    cs_streams_t& s = *sptr;

    for(auto part_streams : s) {
        for(auto css : part_streams) {
            cudaStreamDestroy(css.d2h);
            for(auto s : css.compute) {
                cudaStreamDestroy(s);
            }
        }
    }

    delete sptr;

}


/****************************************************************************

The main function.

*****************************************************************************/

// Forward declaration
void create_switcher_cuda_main(
    std::vector<torch::Tensor>& a,
    std::vector<std::vector<std::vector<torch::Tensor>>>& ksk,
    std::vector<std::vector<torch::Tensor>>& switcher,
    size_t params, size_t streams
);

void create_switcher_cuda(
    // The target of the switcher.
    std::vector<torch::Tensor> a,

    // The ksk.
    std::vector<std::vector<std::vector<torch::Tensor>>> ksk,

    // The output.
    // switcher must contain special primes, such that
    // it will be shrunk down in the calling function.
    std::vector<std::vector<torch::Tensor>> switcher,

    // Params.
    size_t params,

    // Preallocated streams.
    size_t streams
) {
    create_switcher_cuda_main(
        a, ksk, switcher, params, streams
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("create_switcher_cuda", &create_switcher_cuda, "CREATE KEY SWITCHER.");
    m.def("create_cs_params_ptr", &create_cs_params_ptr, "CREATE SWITCHER PARAMS PTR CREATION.");
    m.def("delete_cs_params_ptr", &delete_cs_params_ptr, "CREATE SWITCHER PARAMS PTR DELETION.");
    m.def("preallocate_streams", &preallocate_streams, "PREALLOCATE STREAMS.");
    m.def("destroy_streams", &destroy_streams, "DESTROY STREAMS.");
}
