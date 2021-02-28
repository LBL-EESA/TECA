#ifndef teca_apply_binary_mask_h
#define teca_apply_binary_mask_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_apply_binary_mask)

/// Applies a mask to a given list of variables
/**

Given a mask variable, this routine applies the mask to a list of input
variables.

The mask variable can either be binary, or it can represent a probability
ranging from 0 to 1. For mask variable `mask` and input variable `var`, this
algorithm computes `mask * var` and sends the resulting array downstream; this
masking operation is applied for all variables in the input list.

A potential use-case for this algorithm is masking quantities like
precipitation by the probability of atmospheric river presence; the average
of this masked precipitation variable gives the average precipitation due to
atmospheric rivers.

The output variable names are given a prefix to distinguish them from the
upstream versions. E.g., if the algorithm property `output_var_prefix` is set
to 'ar_', and the variable being masked is 'precip', then the output array
name is 'ar_precip'.

*/
class teca_apply_binary_mask : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_apply_binary_mask)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_apply_binary_mask)
    TECA_ALGORITHM_CLASS_NAME(teca_apply_binary_mask)
    ~teca_apply_binary_mask();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name of the output array
    TECA_ALGORITHM_PROPERTY(std::string, mask_variable)

    // the arrays to mask. if empty no arrays will be
    // requested, and no variables will be masked
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, input_variable)

    // the names of the arrays to store the masking results in
    TECA_ALGORITHM_PROPERTY(std::string, output_var_prefix)

    // adds output_var_prefix to a given variable name
    std::string get_output_variable_name(std::string input_var);

protected:
    teca_apply_binary_mask();

private:
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string mask_variable;
    std::vector<std::string> input_variables;
    std::string output_var_prefix;
};

#endif
