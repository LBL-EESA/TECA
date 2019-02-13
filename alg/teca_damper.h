#ifndef teca_damper_h
#define teca_damper_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_damper)

/**
an algorithm that applies an upside-down gaussian filter to 
damp out the tropics.
To be later implemented: sigma (filter_lat_width_value)
is currently specified directly. Later filter_lat_width_value 
will hold half width at half max (HWHM) as it's more intuitive
than using sigma directly. sigma = hwhm/(sqrt(2*ln(2))).
*/
class teca_damper : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_damper)
    ~teca_damper();

    // Set delta-Y gaussian filter
    TECA_ALGORITHM_PROPERTY(double, filter_lat_width_value)

    // set the names of the arrays that the filter will apply on
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, damper_variable)

    // Set post-fix to be assigned to each input damper variable
    TECA_ALGORITHM_PROPERTY(std::string, post_fix)

protected:
    teca_damper();

    std::vector<std::string> get_damper_variables(const teca_metadata &request);

private:
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    double filter_lat_width_value;
    std::vector<std::string> damper_variables;
    std::string post_fix;
};

#endif
