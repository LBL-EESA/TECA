#ifndef teca_l2_norm_h
#define teca_l2_norm_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_l2_norm)

/// an algorithm that computes L2 norm
/**
Compute L2 norm
*/
class teca_l2_norm : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_l2_norm)
    ~teca_l2_norm();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the arrays that contain the vector components
    // to compute norm from
    TECA_ALGORITHM_PROPERTY(std::string, component_0_variable)
    TECA_ALGORITHM_PROPERTY(std::string, component_1_variable)
    TECA_ALGORITHM_PROPERTY(std::string, component_2_variable)

    // set the name of the array to store the result in.
    // the default is "l2_norm"
    TECA_ALGORITHM_PROPERTY(std::string, l2_norm_variable)

protected:
    teca_l2_norm();

    std::string get_component_0_variable(const teca_metadata &request);
    std::string get_component_1_variable(const teca_metadata &request);
    std::string get_component_2_variable(const teca_metadata &request);
    std::string get_l2_norm_variable(const teca_metadata &request);

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
    std::string component_2_variable;
    std::string l2_norm_variable;
};

#endif
