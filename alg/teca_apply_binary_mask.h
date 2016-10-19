#ifndef teca_apply_binary_mask_h
#define teca_apply_binary_mask_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_apply_binary_mask)

/// an algorithm that applies a binary mask multiplicatively
/**
an algorithm that applies a binary mask multiplicatively to all
arrays in the input dataset. where mask is 1 values are passed
through, where mask is 0 values are removed.
*/
class teca_apply_binary_mask : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_apply_binary_mask)
    ~teca_apply_binary_mask();

    // set the name of the output array
    TECA_ALGORITHM_PROPERTY(std::string, mask_variable)

    // set the arrays to mask. if empty no arrays will be
    // requested, but all present will be masked
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, mask_array)

protected:
    teca_apply_binary_mask();

private:
    //teca_metadata get_output_metadata(unsigned int port,
    //    const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string mask_variable;
    std::vector<std::string> mask_arrays;
};

#endif
