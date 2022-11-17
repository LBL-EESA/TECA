#include "teca_binary_segmentation_internals.h"
#include "teca_config.h"

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
    p_teca_char_array segmentation = teca_char_array::New(n_elem);

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        input_array.get(),

        auto sp_in = static_cast<const TT*>(input_array.get())->get_cpu_accessible();
        const NT *p_in = sp_in.get();

        auto sp_seg = segmentation->get_cpu_accessible();
        char *p_seg = sp_seg.get();

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
