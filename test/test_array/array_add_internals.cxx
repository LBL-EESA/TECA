#include "array_add_internals.h"
#include "array.h"

namespace array_add_internals
{
// **************************************************************************
int cpu_dispatch(p_array &result, const const_p_array &array_1,
    const const_p_array &array_2, size_t n_vals)
{
#ifndef TECA_NDEBUG
    std::cerr << teca_parallel_id()
        << "array_add_internals::cpu_dispatch" << std::endl;
#endif

    // make sure that inputs are on the CPU
    std::shared_ptr<const double> parray_1 = array_1->get_host_accessible();
    std::shared_ptr<const double> parray_2 = array_2->get_host_accessible();

    // allocate the result on the CPU
    result = array::new_host_accessible();
    result->resize(n_vals);

    array_add_internals::cpu::add(result->data(),
        parray_1.get(), parray_2.get(), n_vals);

    return 0;
}

#if !defined(TECA_HAS_CUDA)
// **************************************************************************
int cuda_dispatch(int device_id, p_array &result,
    const const_p_array &array_1, const const_p_array &array_2,
    size_t n_vals)
{
    (void) device_id;
    (void) result;
    (void) array_1;
    (void) array_2;
    (void) n_vals;

    TECA_ERROR("array_add failed because CUDA is not available")

    return -1;
}
#endif
}
