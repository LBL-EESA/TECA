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
 * Given a mask variable, this routine applies the mask to a list of input
 * variables.
 *
 * The mask variable can either be binary, or it can represent a probability
 * ranging from 0 to 1. For mask variable `mask` and input variable `var`, this
 * algorithm computes `mask * var` and sends the resulting array downstream; this
 * masking operation is applied for all variables in the input list.
 *
 * A potential use-case for this algorithm is masking quantities like
 * precipitation by the probability of atmospheric river presence; the average
 * of this masked precipitation variable gives the average precipitation due to
 * atmospheric rivers.
 *
 * The output variable names are given a prefix to distinguish them from the
 * upstream versions. E.g., if the algorithm property `output_variable_prefix` is set
 * to 'ar_', and the variable being masked is 'precip', then the output array
 * name is 'ar_precip'.
 */
class TECA_EXPORT teca_apply_binary_mask : public teca_algorithm
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

    /** @name mask_variable
     * set the name of the variable containing the mask values
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, mask_variable)
    ///@}

    /** @name masked_variable
     * A list of of variables to apply the mask to. If empty no arrays will be
     * requested, and no variables will be masked
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, masked_variable)
    ///@}

    /** @name output_variable_prefix
     * A prefix for the names of the variables that have been masked.  If this
     * is empty masked data replaces its input, otherwise input data is
     * preserved and masked data is added.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, output_variable_prefix)
    ///@}

    /** helper that constructs and returns the result variable names taking
     * into account he list of masked_variables and the output_variable_prefix.
     * use this to know what variables will be produced.
     */
    void get_output_variable_names(std::vector<std::string> &names);

protected:
    teca_apply_binary_mask();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    // helper that given and input variable name constructs the result variable
    // name taking into account the output_variable_prefix
    std::string get_output_variable_name(std::string input_var);

private:
    std::string mask_variable;
    std::vector<std::string> masked_variables;
    std::string output_variable_prefix;
};

#endif
