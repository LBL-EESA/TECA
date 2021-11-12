#ifndef teca_latitude_damper_h
#define teca_latitude_damper_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_latitude_damper)

/// Inverted Gaussian damper for scalar fields.
/**
 * Damps the specified scalar field(s) using an inverted Gaussian centered on a
 * given latitude with a half width specified in degrees latitude. The
 * parameters defining the Gaussian (center, half width at half max) can be
 * specified by the user directly or by down stream algorithm via the
 * following keys in the request.
 *
 * request keys:
 *
 *   teca_latitude_damper::damped_variables
 *   teca_latitude_damper::half_width_at_half_max
 *   teca_latitude_damper::center
 *
 * @note
 * User specified values take precedence over request keys. When
 * using request keys be sure to include the variable post-fix.
 */
class TECA_EXPORT teca_latitude_damper : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_latitude_damper)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_latitude_damper)
    TECA_ALGORITHM_CLASS_NAME(teca_latitude_damper)
    ~teca_latitude_damper();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the center of the Gaussian in units of degrees latitude.
    // default is 0.0 deg lat
    TECA_ALGORITHM_PROPERTY(double, center)

    // set the half width of the Gaussian in units of degrees latitude.
    // default is 45.0 deg lat
    TECA_ALGORITHM_PROPERTY(double, half_width_at_half_max)

    // set the names of the arrays that the filter will apply on
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, damped_variable)

    // a string to be appended to the name of each output variable
    // setting this to an empty string will result in the damped array
    // replacing the input array in the output. default is an empty
    // string ""
    TECA_ALGORITHM_PROPERTY(std::string, variable_postfix)

protected:
    teca_latitude_damper();

    // helpers to get parameters defining the Gaussian used by the
    // the filter. if the user has not specified a value then the
    // request is probed. a return of zero indicates success
    int get_sigma(const teca_metadata &request, double &sigma);
    int get_mu(const teca_metadata &request, double &mu);

    // helper to get the list of variables to apply the filter on
    // if the user provided none, then the request is probed. a
    // return of 0 indicates success
    int get_damped_variables(std::vector<std::string> &vars);

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

private:
    double center;
    double half_width_at_half_max;
    std::vector<std::string> damped_variables;
    std::string variable_postfix;
};

#endif
