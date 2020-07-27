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
    TECA_ALGORITHM_PROPERTY(std::string, long_name)
    TECA_ALGORITHM_PROPERTY(std::string, units)
    TECA_ALGORITHM_PROPERTY(std::string, pressure_level_variable)
    TECA_ALGORITHM_PROPERTY(std::string, integration_variable)
    TECA_ALGORITHM_PROPERTY(std::string, output_variable_name)

protected:
    teca_vertical_integral();

private:

    std::string long_name;
    std::string units;
    std::string pressure_level_variable;
    std::string integration_variable;
    std::string output_variable_name;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void get_dependent_variables(const teca_metadata &request,
        std::vector<std::string> &dep_vars);

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;
};

#endif
