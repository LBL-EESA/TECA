#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_cuda_util.h"
#include "teca_programmable_algorithm.h"
#include "teca_metadata.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <iostream>

#include "teca_cartesian_mesh.h"

using namespace teca_variant_array_util;
using allocator = teca_variant_array::allocator;


// **************************************************************************
template<typename NT>
__global__
void initialize_cuda(NT *data, double val, size_t n_vals)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    data[i] = val;
}

// **************************************************************************
template <typename NT, typename TT = teca_variant_array_impl<NT>>
std::shared_ptr<TT> initialize_cuda(size_t n_vals, const NT &val)
{
    // allocate the memory
    auto [ao, pao] = ::New<TT>(n_vals, allocator::cuda_async);

    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch parameters" << std::endl;
        return nullptr;
    }

    // initialize the data
    cudaError_t ierr = cudaSuccess;
    initialize_cuda<<<block_grid, thread_grid>>>(pao, val, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the print kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    std::cerr << "initialized to an array of " << n_vals << " elements of type "
        << typeid(NT).name() << sizeof(NT) << " to " << val << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ao = "; ao->debug_print(); std::cerr << std::endl;
        ao->debug_print();
        std::cerr << std::endl;
    }

    //cudaDeviceSynchronize();

    return ao;
}






// **************************************************************************
template<typename NT1, typename NT2>
__global__
void add_cuda(NT1 *result, const NT1 *array_1, const NT2 *array_2, size_t n_vals)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    result[i] = array_1[i] + array_2[i];
}

// **************************************************************************
template <typename NT1, typename NT2>
p_teca_variant_array_impl<NT1> add_cuda(const const_p_teca_variant_array_impl<NT1> &a1,
    const const_p_teca_variant_array_impl<NT2> &a2)
{
    using TT1 = teca_variant_array_impl<NT1>;
    using TT2 = teca_variant_array_impl<NT2>;

    // get the inputs
    auto [spa1, pa1] = get_cuda_accessible<TT1>(a1);
    auto [spa2, pa2] = get_cuda_accessible<TT2>(a2);

    // allocate the memory
    size_t n_vals = a1->size();
    auto [ao, pao] = ::New<TT1>(n_vals, NT1(0), allocator::cuda_async);

    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch parameters" << std::endl;
        return nullptr;
    }

    // initialize the data
    cudaError_t ierr = cudaSuccess;
    add_cuda<<<block_grid, thread_grid>>>(pao, pa1, pa2, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the print kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    std::cerr << "added array of " << n_vals << " elements of type "
        << typeid(NT1).name() << sizeof(NT1) << " to array of type "
        << typeid(NT2).name() << sizeof(NT2) << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "a1 = "; a1->debug_print(); std::cerr << std::endl;
        std::cerr << "a2 = "; a2->debug_print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao->debug_print(); std::cerr << std::endl;
    }

    //cudaDeviceSynchronize();

    return ao;
}





// **************************************************************************
template<typename NT1, typename NT2>
__global__
void multiply_scalar_cuda(NT1 *result, const NT1 *array_in, NT2 scalar, size_t n_vals)
{
    unsigned long i = teca_cuda_util::thread_id_to_array_index();

    if (i >= n_vals)
        return;

    result[i] = array_in[i] * scalar;
}

// **************************************************************************
template <typename NT1, typename NT2>
p_teca_variant_array_impl<NT1> multiply_scalar_cuda(
    const const_p_teca_variant_array_impl<NT1> &ain, const NT2 &val)
{
    using TT1 = teca_variant_array_impl<NT1>;

    // get the inputs
    auto [spain, pain] = get_cuda_accessible<TT1>(ain);

    // allocate the memory
    size_t n_vals = ain->size();
    auto [ao, pao] = ::New<TT1>(n_vals, NT1(0), allocator::cuda_async);

    // determine kernel launch parameters
    int n_blocks = 0;
    dim3 block_grid;
    dim3 thread_grid;
    if (teca_cuda_util::partition_thread_blocks(0, n_vals,
        8, block_grid, n_blocks, thread_grid))
    {
        std::cerr << "ERROR: Failed to determine launch parameters" << std::endl;
        return nullptr;
    }

    // initialize the data
    cudaError_t ierr = cudaSuccess;
    multiply_scalar_cuda<<<block_grid, thread_grid>>>(pao, pain, val, n_vals);
    if ((ierr = cudaGetLastError()) != cudaSuccess)
    {
        std::cerr << "ERROR: Failed to launch the print kernel. "
            << cudaGetErrorString(ierr) << std::endl;
        return nullptr;
    }

    std::cerr << "multiply_scalar " << val << " type "
        << typeid(NT2).name() << sizeof(NT2) << " by array type "
        << typeid(NT1).name() << sizeof(NT1) << " with " << n_vals
        << " elements" << std::endl;

    if (n_vals < 33)
    {
        std::cerr << "ain = "; ain->debug_print(); std::cerr << std::endl;
        std::cerr << "ao = "; ao->debug_print(); std::cerr << std::endl;
    }

    //cudaDeviceSynchronize();

    return ao;
}



// **************************************************************************
template <typename NT>
int compare_int(const const_p_teca_variant_array_impl<NT> &ain, int val)
{
    size_t n_vals = ain->size();

    std::cerr << "comparing array with " << n_vals
        << " elements to " << val << std::endl;

    p_teca_int_array ai = teca_int_array::New(n_vals, ain->get_allocator());
    ain->get(ai);

    if (n_vals < 33)
    {
        ai->debug_print();
    }

    auto [spai, pai] = get_host_accessible<teca_int_array>(ai);

    sync_host_access_any(ai);

    for (size_t i = 0; i < n_vals; ++i)
    {
        if (pai[i] != val)
        {
            std::cerr << "ERROR: pai[" << i << "] = " << pai[i]
                << " != " << val << std::endl;
            return -1;
        }
    }

    std::cerr << "all elements are equal to " << val << std::endl;

    return 0;
}



int main(int, char **)
{
    allocator cuda_alloc = allocator::cuda_async;
    allocator cpu_alloc = allocator::malloc;

    size_t n_vals = 100000;

    std::string array_name = "test_array";
    double initial_value = 3.14;
    double add_value = 0.001592;
    double mult_value = 1.0e6;
    double comp_value = 3141592;
    int test_status = -1;

    // this stage is a source producing a CUDA and CPU array
    p_teca_programmable_algorithm array_src = teca_programmable_algorithm::New();
    array_src->set_number_of_input_connections(0);
    array_src->set_report_callback([](unsigned int port,
        const std::vector<teca_metadata> &md_in) -> teca_metadata
        {
            (void) port;
            (void) md_in;

            // setup the execution control keys
            teca_metadata md_out;
            md_out.set("index_initializer_key", std::string("num_passes"));
            md_out.set("num_passes", 1l);
            md_out.set("index_request_key", std::string("pass_no"));

            return md_out;
        });
    array_src->set_execute_callback(
        [&](unsigned int port,
            const std::vector<const_p_teca_dataset> &input_data,
            const teca_metadata &request) -> const_p_teca_dataset
        {
            (void) port;
            (void) input_data;
            (void) request;

            std::cerr << "initializing to " << initial_value << std::endl;

            p_teca_float_array  cuda_out =
                 teca_float_array::New(n_vals, initial_value, cuda_alloc);  // (CUDA)

            p_teca_float_array  cpu_out =
                 teca_float_array::New(n_vals, initial_value, cpu_alloc);  // (CPU)

            // construct the output
            auto col_out = teca_array_collection::New();
            col_out->set("cuda_array", cuda_out);
            col_out->set("cpu_array", cpu_out);

            return col_out;
        });

    // this stage adds a scalar value to the CUDA and CPU arrays
    p_teca_programmable_algorithm array_add = teca_programmable_algorithm::New();
    array_add->set_input_connection(array_src->get_output_port());
    array_add->set_execute_callback(
        [&](unsigned int port,
            const std::vector<const_p_teca_dataset> &input_data,
            const teca_metadata &request) -> const_p_teca_dataset
        {
            (void) port;
            (void) request;

            std::cerr << "adding " << initial_value << " + " << add_value
                << " = "  << initial_value + add_value << std::endl;

            const_p_teca_array_collection col_in =
                std::dynamic_pointer_cast<const teca_array_collection>
                    (input_data[0]);

            const_p_teca_variant_array cuda_in = col_in->get("cuda_array");
            const_p_teca_variant_array cpu_in = col_in->get("cpu_array");

            size_t n_elem = cuda_in->size();

            p_teca_variant_array add_val = cuda_in->new_instance(cuda_alloc);

            p_teca_variant_array cuda_out;
            p_teca_variant_array cpu_out;

            VARIANT_ARRAY_DISPATCH(add_val.get(),

                // add data already acccessible in CUDA
                auto pcuda_in = std::dynamic_pointer_cast<const TT>(cuda_in);

                auto padd_val = std::static_pointer_cast<TT>(add_val);
                padd_val->resize(n_elem, add_value);

                cuda_out = add_cuda(pcuda_in, const_ptr(padd_val));

                // add data accessible in the CPU
                auto pcpu_in = std::dynamic_pointer_cast<const TT>(cpu_in);
                auto tmp = add_cuda(pcpu_in, const_ptr(padd_val));

                // move back to the CPU
                auto pcpu_out = TT::New(cpu_alloc);
                pcpu_out->assign(const_ptr(tmp));
                cpu_out = pcpu_out;
                )

            // construct the output
            p_teca_array_collection col_out =
                std::static_pointer_cast<teca_array_collection>
                    (col_in->new_instance());

            col_out->set("cuda_array", cuda_out);
            col_out->set("cpu_array", cpu_out);

            return col_out;
        });


    // this stage multiplies a scalar value to the CUDA and CPU arrays
    p_teca_programmable_algorithm array_mult = teca_programmable_algorithm::New();
    array_mult->set_input_connection(array_add->get_output_port());
    array_mult->set_execute_callback(
        [&](unsigned int port,
            const std::vector<const_p_teca_dataset> &input_data,
            const teca_metadata &request) -> const_p_teca_dataset
        {
            (void) port;
            (void) request;

            std::cerr << "multiplying " << mult_value << " * ("
                << initial_value << " + " << add_value << ") = "
                << mult_value * (initial_value + add_value) << std::endl;

            const_p_teca_array_collection col_in =
                std::dynamic_pointer_cast<const teca_array_collection>
                    (input_data[0]);

            const_p_teca_variant_array cuda_in = col_in->get("cuda_array");
            const_p_teca_variant_array cpu_in = col_in->get("cpu_array");

            p_teca_variant_array cuda_out;
            p_teca_variant_array cpu_out;

            VARIANT_ARRAY_DISPATCH(cuda_in.get(),

                // mult data already acccessible in CUDA
                auto pcuda_in = std::dynamic_pointer_cast<const TT>(cuda_in);
                cuda_out = multiply_scalar_cuda(pcuda_in, mult_value);

                // mult data accessible in the CPU
                auto pcpu_in = std::dynamic_pointer_cast<const TT>(cpu_in);
                auto tmp = multiply_scalar_cuda(pcpu_in, mult_value);

                // move back to the CPU
                auto pcpu_out = tmp->new_instance(cpu_alloc);
                pcpu_out->assign(const_ptr(tmp));
                cpu_out = pcpu_out;
                )

            // construct the output
            p_teca_array_collection col_out =
                std::static_pointer_cast<teca_array_collection>
                    (col_in->new_instance());

            col_out->set("cuda_array", cuda_out);
            col_out->set("cpu_array", cpu_out);
            return col_out;
        });

    // this stage compares the CUDA and CPU arrays to the expected value
    p_teca_programmable_algorithm array_comp = teca_programmable_algorithm::New();
    array_comp->set_input_connection(array_mult->get_output_port());
    array_comp->set_execute_callback(
        [&](unsigned int port,
            const std::vector<const_p_teca_dataset> &input_data,
            const teca_metadata &request) -> const_p_teca_dataset
        {
            (void) port;
            (void) request;

            std::cerr << "compare to " << comp_value << std::endl;

            test_status = -1;

            // get the input
            const_p_teca_array_collection col_in =
                std::dynamic_pointer_cast<const teca_array_collection>
                    (input_data[0]);

            const_p_teca_variant_array cuda_in = col_in->get("cuda_array");
            const_p_teca_variant_array cpu_in = col_in->get("cpu_array");


            VARIANT_ARRAY_DISPATCH(cuda_in.get(),

                // compare data already acccessible in CUDA
                auto pcuda_in = std::dynamic_pointer_cast<const TT>(cuda_in);
                if (compare_int(pcuda_in, comp_value))
                {
                    TECA_ERROR("The CUDA array was not equal to the expected"
                        " value " << comp_value);
                    if (pcuda_in->size() < 33)
                        pcuda_in->debug_print();
                    return nullptr;
                }

                // compare data already acccessible in CPU
                auto pcpu_in = std::dynamic_pointer_cast<const TT>(cpu_in);
                if (compare_int(pcpu_in, comp_value))
                {
                    TECA_ERROR("The CPU array was not equal to the expected"
                        " value " << comp_value);
                    if (pcpu_in->size() < 33)
                        pcpu_in->debug_print();
                    return nullptr;
                }
                )

            test_status = 0;
            TECA_STATUS("The arrays had the expected value " << comp_value)
            return col_in;
        });

    // run the pipeline
    array_comp->update();

    return test_status;
}
