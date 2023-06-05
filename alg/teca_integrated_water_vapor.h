#ifndef teca_integrated_water_vapor_h
#define teca_integrated_water_vapor_h

#include "teca_shared_object.h"
#include "teca_vertical_reduction.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_integrated_water_vapor)

/// An algorithm that computes integrated water vapor (IWV)
/**
 * Compute column integrated water vapor (IWV) from the specific humidity.
 *
 * \f[
 * IWV = \frac{1}{g} \int_{p_{sfc}}^{p_{top}} q dp
 * \f]
 *
 * where \f$q\f$ is the specific humidity.
 *
 * This calculation is an instance of a vertical reduction where
 * a 3D mesh is transformed into a 2D one.
 *
 * This algorithm handles missing values.
 */
class TECA_EXPORT teca_integrated_water_vapor : public teca_vertical_reduction
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_integrated_water_vapor)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_integrated_water_vapor)
    TECA_ALGORITHM_CLASS_NAME(teca_integrated_water_vapor)
    ~teca_integrated_water_vapor();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name specific_humidity_variable
     * set the name of the variable that contains the specific humidity ("hus")
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, specific_humidity_variable)
    ///@}

    /** @name iwv_variable
     * set the name of the varaiable that contains the integrated water vapor
     * ("iwv").
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, iwv_variable)
    ///@}

    /** @name fill_value
     * set the _fillValue attribute for the output data.  default 1.0e20
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, fill_value)
    ///@}

protected:
    teca_integrated_water_vapor();

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
    std::string specific_humidity_variable;
    std::string iwv_variable;
    double fill_value;
};

#endif
