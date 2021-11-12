#ifndef teca_l2_norm_h
#define teca_l2_norm_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_l2_norm)

/// An algorithm that computes L2 norm
class TECA_EXPORT teca_l2_norm : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_l2_norm)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_l2_norm)
    TECA_ALGORITHM_CLASS_NAME(teca_l2_norm)
    ~teca_l2_norm();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name component_0_variable
     * Set the arrays that contain the vector components to compute the norm
     * from.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, component_0_variable)
    ///@}

    /** @name component_1_variable
     * Set the arrays that contain the vector components to compute the norm
     * from.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, component_1_variable)
    ///@}

    /** @name component_2_variable
     * Set the arrays that contain the vector components to compute the norm
     * from.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, component_2_variable)
    ///@}

    /** @name l2_norm_variable
     * set the name of the array to store the result in.  the default is
     * "l2_norm"
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, l2_norm_variable)
    ///@}

protected:
    teca_l2_norm();
/*
    // helpers to get the variable names from either the incoming
    // request or the class member variable.
    std::string get_component_0_variable(const teca_metadata &request);
    std::string get_component_1_variable(const teca_metadata &request);
    std::string get_component_2_variable(const teca_metadata &request);
    std::string get_l2_norm_variable(const teca_metadata &request);
*/
private:
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
    std::string component_0_variable;
    std::string component_1_variable;
    std::string component_2_variable;
    std::string l2_norm_variable;
};

#endif
