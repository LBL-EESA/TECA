#ifndef teca_vertical_integral_h
#define teca_vertical_integral_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_vertical_integral)

/// compute statistics about connected components
class teca_vertical_integral : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_vertical_integral)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_vertical_integral)
    TECA_ALGORITHM_CLASS_NAME(teca_vertical_integral)
    ~teca_vertical_integral();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name of the the arrays involved in integration
    TECA_ALGORITHM_PROPERTY(std::string, hybrid_a_variable)
    TECA_ALGORITHM_PROPERTY(std::string, hybrid_b_variable)
    TECA_ALGORITHM_PROPERTY(std::string, sigma_variable)
    TECA_ALGORITHM_PROPERTY(std::string, surface_p_variable)
    TECA_ALGORITHM_PROPERTY(std::string, p_top_variable)
    TECA_ALGORITHM_PROPERTY(std::string, integration_variable)
    TECA_ALGORITHM_PROPERTY(std::string, output_variable_name)
    // set whether the vertical coordinate is hybrid or sigma
    TECA_ALGORITHM_PROPERTY(int, using_hybrid)
    TECA_ALGORITHM_PROPERTY(float, p_top_override_value)

protected:
    teca_vertical_integral();

private:

    std::string hybrid_a_variable;
    std::string hybrid_b_variable;
    std::string sigma_variable;
    std::string surface_p_variable;
    std::string p_top_variable;
    std::string integration_variable;
    std::string output_variable_name;
    int using_hybrid = true;
    float p_top_override_value = -1;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void get_dependent_variables(const teca_metadata &request,
        std::vector<std::string> &dep_vars);
};

#endif
