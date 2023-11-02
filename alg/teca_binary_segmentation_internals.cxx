#include "teca_binary_segmentation_internals.h"
#include "teca_config.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"

using namespace teca_variant_array_util;


namespace teca_binary_segmentation_internals
{
#if !defined(TECA_HAS_CUDA)
// **************************************************************************
int cuda_dispatch(int device_id,
    p_teca_variant_array &output_array,
    const const_p_teca_variant_array &input_array,
    int threshold_mode,
    double low, double high)
{
    (void)device_id;
    (void)output_array;
    (void)input_array;
    (void)threshold_mode;
    (void)low;
    (void)high;

    TECA_ERROR("CUDA support is not available")
    return -1;
}
#endif

// **************************************************************************
int cpu_dispatch(
    p_teca_variant_array &output_array,
    const const_p_teca_variant_array &input_array,
    int threshold_mode,
    double low, double high)
{
    // do segmentation
    size_t n_elem = input_array->size();
    auto [segmentation, p_seg] = ::New<teca_char_array>(n_elem);

    VARIANT_ARRAY_DISPATCH(input_array.get(),

        auto [sp_in, p_in] = get_host_accessible<CTT>(input_array);

        sync_host_access_any(input_array);

        if (threshold_mode == teca_binary_segmentation::BY_VALUE)
        {
            cpu::value_threshold(p_seg, p_in, n_elem,
               static_cast<NT>(low), static_cast<NT>(high));
        }
        else if (threshold_mode == teca_binary_segmentation::BY_PERCENTILE)
        {
            cpu::percentile_threshold(p_seg, p_in, n_elem,
                static_cast<NT>(low), static_cast<NT>(high));
        }
        else
        {
            TECA_ERROR("Invalid threshold mode")
            return -1;
        }
        )

    output_array = segmentation;
    return 0;
}
}
