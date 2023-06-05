#ifndef teca_integrated_vapor_transport_h
#define teca_integrated_vapor_transport_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_vertical_reduction.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_integrated_vapor_transport)

/// An algorithm that computes integrated vapor transport (IVT)
/**
 * Compute integrated vapor transport (IVT) from wind vector and
 * specific humidity.
 *
 * \f[
 * \vec{IVT} = \frac{1}{g} \int_{p_{sfc}}^{p_{top}} \vec{v} \, q \; dp
 * \f]
 *
 * where \f$q\f$ is the specific humidity, and \f$\vec{v} = (u, v)\f$ are the
 * longitudinal and latitudinal components of wind.
 *
 * This calculation is an instance of a vertical reduction where
 * a 3D mesh is transformed into a 2D one.
 *
 * This algorithm handles missing values.
 */
class TECA_EXPORT teca_integrated_vapor_transport : public teca_vertical_reduction
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_integrated_vapor_transport)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_integrated_vapor_transport)
    TECA_ALGORITHM_CLASS_NAME(teca_integrated_vapor_transport)
    ~teca_integrated_vapor_transport();

    /** @name program_options
     * report/initialize to/from Boost program options objects.
     */
    ///@{
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()
    ///@}

    /** @name wind_u_variable
     * set the name of the varaiable that contains the longitudinal component
     * of the wind vector ("ua")
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, wind_u_variable)
    ///@}

    /** @name wind_v_variable
     * set the name of the varaiable that contains the latitudinal component of
     * the wind vector ("va")
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, wind_v_variable)
    ///@}

    /** @name specific_humidity_variable
     * set the name of the variable that contains the specific humidity ("hus")
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, specific_humidity_variable)
    ///@}

    /** @name ivt_u_variable
     * set the name of the varaiable that contains the longitudinal component
     * of the ivt vector ("ivt_u")
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, ivt_u_variable)
    ///@}

    /** @name ivt_v_variable
     * set the name of the varaiable that contains the latitudinal component of
     * the ivt vector ("ivt_v")
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, ivt_v_variable)
    ///@}

    /** @name fill_value
     * set the _fillValue attribute for the output data.  default 1.0e20
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, fill_value)
    ///@}

    /** @name surface_pressure
     * set the surface pressure, default 101235. only used when not using
     * trapezoid integration. See set_use_trapezoid_rule.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, surface_pressure)
    ///@}

    /** @name top_pressure
     * set the top pressure, default 0. only used when not using
     * trapezoid integration. See set_use_trapezoid_rule.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, top_pressure)
    ///@}

    /** @name use_trapezoid_rule
     * if set then trapezoid rule is used for integration. This is a higher
     * order scheme than the default but the pressure level thickness is
     * calculated between cell centers.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, use_trapezoid_rule)
    ///@}

protected:
    teca_integrated_vapor_transport();

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
    std::string wind_u_variable;
    std::string wind_v_variable;
    std::string specific_humidity_variable;
    std::string ivt_u_variable;
    std::string ivt_v_variable;
    double fill_value;
    double surface_pressure;
    double top_pressure;
    int use_trapezoid_rule;
};

#endif
