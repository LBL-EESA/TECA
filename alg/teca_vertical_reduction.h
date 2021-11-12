#ifndef teca_vertical_reduction_h
#define teca_vertical_reduction_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_vertical_reduction)

/// The base class for vertical reducitons.
/**
 * implements common operations associated with computing a vertical
 * reduction where a 3D dataset is transformed into a 2D dataset
 * by a reduction along the 3rd spatial dimension.
*/
class TECA_EXPORT teca_vertical_reduction : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_vertical_reduction)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_vertical_reduction)
    TECA_ALGORITHM_CLASS_NAME(teca_vertical_reduction)
    ~teca_vertical_reduction();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name dependent_variable
     * set/get the list of variables that are needed to produce the derived
     * quantity
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, dependent_variable)
    ///@}

    /** @name derived_variable
     * set/get the name of the variable that is produced
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, derived_variable)
    ///@}

    /** @name derived_variable_attribute
     * Set the attributes of the variable that is produced.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(teca_array_attributes, derived_variable_attribute)
    ///@}

protected:
    teca_vertical_reduction();

    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::vector<std::string> dependent_variables;
    std::vector<std::string> derived_variables;
    std::vector<teca_array_attributes> derived_variable_attributes;
};

#endif
