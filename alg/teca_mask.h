#ifndef teca_mask_h
#define teca_mask_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_mask)

/// an algorithm that masks a range of values
/**
An algorithm to mask a range of values in an array. Values
in the range are replaced with the mask value.
*/
class teca_mask : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_mask)
    ~teca_mask();

    // set the names of the arrays to apply the mask to
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, mask_variable)

    // Set the range identifying values to mask. Values inside
    // this range are masked.  The defaults are (-infinity, infinity].
    TECA_ALGORITHM_PROPERTY(double, low_threshold_value)
    TECA_ALGORITHM_PROPERTY(double, high_threshold_value)

    // Set the value used to replace input values that
    // are inside the specified range.
    TECA_ALGORITHM_PROPERTY(double, mask_value)

protected:
    teca_mask();

    std::vector<std::string> get_mask_variables(
        const teca_metadata &request);

private:
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::vector<std::string> mask_variables;
    double low_threshold_value;
    double high_threshold_value;
    double mask_value;
};

#endif
