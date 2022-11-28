#ifndef teca_valid_value_mask_h
#define teca_valid_value_mask_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

/// @name mask type aliases
///@{
using NT_MASK = char;
using TT_MASK = teca_variant_array_impl<NT_MASK>;
using CTT_MASK = teca_variant_array_impl<NT_MASK>;
using PT_MASK = p_teca_variant_array_impl<NT_MASK>;
using CPT_MASK = const_p_teca_variant_array_impl<NT_MASK>;
using SP_MASK = std::shared_ptr<NT_MASK>;
using CSP_MASK = std::shared_ptr<const NT_MASK>;
///@}

TECA_SHARED_OBJECT_FORWARD_DECL(teca_valid_value_mask)

/// an algorithm that computes a mask identifying valid values
/**
 * For each requested mask, from its associated input array, compute a mask set to
 * 1 where the data is valid and 0 everywhere else.  Downstream algorithms then
 * may look for the mask array and process the data in such a way as to produce
 * valid results in the presence of missing data.
 *
 * Validity is determined by comparing the array's elements to the fill value as
 * specified in the array's attributes _FillValue or missing_value field.  If
 * neither of these attribute fields are present then no mask is computed.
 *
 * The masks generated are stored in the output mesh with the same centering as
 * the input variable they were generated from, and named using the variable's
 * name with the string "_valid" appended. For example if a mask was generated for
 * a variable named "V" it will be named "V_valid".
 *
 * Masks are requested for specific arrays in one of two ways. One may use the
 * mask_arrays algorithm property to explicitly name the list of variables to
 * compute masks for. Alternatively, a heuristic applied to incoming requests
 * determines if masks should be generated. Specifically the string "_valid" is
 * looked for at the end of each requested array.  If it is found then the mask
 * for the variable named by removing "_valid" is generated.  For example the
 * request for "V_valid" would result in the mask being generated for the variable
 * "V".
*/
class TECA_EXPORT teca_valid_value_mask : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_valid_value_mask)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_valid_value_mask)
    TECA_ALGORITHM_CLASS_NAME(teca_valid_value_mask)
    ~teca_valid_value_mask();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name mask_arrays
     * explicitly set a list of input arrays to process. By default
     * all arrays are processed. Use this property to compute masks
     * for a subset of the arrays,
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, mask_array)
    ///@}

    /** @name enable_valid_range
     * enable the use of valid_range, valid_min, valid_max attributes.
     * by default this is off.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, enable_valid_range)
    ///@}

protected:
    teca_valid_value_mask();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::vector<std::string> mask_arrays;
    int enable_valid_range;
};

#endif
