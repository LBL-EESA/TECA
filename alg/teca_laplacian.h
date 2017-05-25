#ifndef teca_laplacian_h
#define teca_laplacian_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_laplacian)

/// an algorithm that computes laplacian
/**
Compute laplacian from a vector field.
*/
class teca_laplacian : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_laplacian)
    ~teca_laplacian();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the arrays that contain the vector components
    // to compute laplacian from
    TECA_ALGORITHM_PROPERTY(std::string, component_0_variable)
    TECA_ALGORITHM_PROPERTY(std::string, component_1_variable)

    // set the name of the array to store the result in.
    // the default is "laplacian"
    TECA_ALGORITHM_PROPERTY(std::string, laplacian_variable)

protected:
    teca_laplacian();

    std::string get_component_0_variable(const teca_metadata &request);
    std::string get_component_1_variable(const teca_metadata &request);
    std::string get_laplacian_variable(const teca_metadata &request);

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
    std::string laplacian_variable;
};

#endif
