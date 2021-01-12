#ifndef teca_vorticity_h
#define teca_vorticity_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_vorticity)

/// an algorithm that computes vorticity
/**
Compute vorticity from a vector field.
*/
class teca_vorticity : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_vorticity)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_vorticity)
    TECA_ALGORITHM_CLASS_NAME(teca_vorticity)
    ~teca_vorticity();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @anchor component_0_variable
     * @name component_0_variable
     * Set the 1st array that contains the vector components
     * to compute vorticity from
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, component_0_variable)
    ///@}

    /** @anchor component_1_variable
     * @name component_1_variable
     * Set the 2nd array that contains the vector components
     * to compute vorticity from
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, component_1_variable)
    ///@}

    /** @anchor vorticity_variable
     * @name vorticity_variable
     * Set the name of the array to store the result in.
     * the default is "vorticity"
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, vorticity_variable)
    ///@}

protected:
    teca_vorticity();

    std::string get_component_0_variable(const teca_metadata &request);
    std::string get_component_1_variable(const teca_metadata &request);
    std::string get_vorticity_variable(const teca_metadata &request);

private:
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
    std::string component_0_variable;
    std::string component_1_variable;
    std::string vorticity_variable;
};

#endif
