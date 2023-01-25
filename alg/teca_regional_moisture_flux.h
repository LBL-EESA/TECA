#ifndef teca_regional_moisture_flux_h
#define teca_regional_moisture_flux_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_regional_moisture_flux)

/// Computes moisture flux from IVT over a geographic region
/** Computes moisture flux from IVT over a geographic region.
 *
 * The net transport of water vapor \f$F\f$ into/out of the atmosphere above a
 * region enclosed by the closed curve \f$C\f$ is given by:
 *
 * \f[
 *     F = - r_e \iint\limits_{C} \left( \frac{\partial}{\partial\phi} \left[ IVT_{\phi} \; sin \phi \right] + \frac{\partial}{\partial\theta} [ IVT_{\theta} ] \right) \; d\theta \, d\phi
 * \f]
 *
 * where:
 *
 * \f$IVT_{\theta}\f$  :  longitudinal component of the IVT vector in units of \f$kg\, m^{-1} \, s^{-1}\f$ <br>
 * \f$IVT_{\phi}\f$  :  latitudinal component of the IVT vector in units of \f$kg \, m^{-1} \, s^{-1}\f$ <br>
 * \f$\phi\f$  :  \f$(90 -  \f$degrees latitude\f$) \frac{\pi}{180}\f$ radians <br>
 * \f$\theta\f$  :  \f$(\f$degrees longitude\f$) \frac{\pi}{180}\f$ radians <br>
 * \f$C\f$  :  a curve enclosing a region on the surface of the Earth <br>
 * \f$r_e\f$  :  the radius of the Earth (6378100) \f$m\f$ <br>
 *
 * Note: the minus sign out front is because we flip the limits of integration
 * when going from latitude to \f$\phi\f$ coordinates.
 *
 * Requires:
 *  1. IVT vector components in units of \f$kg s^{-1} m^{-1}\f$
 *  2. A 2D mask where the non-zero entries define an arbitrary region of
 *     interest on the surface of the Earth. ::teca_shape_file_mask is an
 *     algorithm that can be used to generate such a mask from an ESRI
 *     shapefile.
 *
 * Produces:
 * A table with a column for the net moisture flux over the region as well as
 * collumns for time and time step.
 *
 * @attention This algorithm is not suitable for general atmospheric flux
 * calculations due to simplifications made in the derivation that only apply
 * when caclulating moisture flux.
 *
 * @attention This algorithm does not currently handle missing values.
 *
 */
class TECA_EXPORT teca_regional_moisture_flux : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_regional_moisture_flux)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_regional_moisture_flux)
    TECA_ALGORITHM_CLASS_NAME(teca_regional_moisture_flux)
    ~teca_regional_moisture_flux();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name ivt_u_variable
     * Set the name of the variable containing the longitudinal component of
     * IVT.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, ivt_u_variable)
    ///@}

    /** @name ivt_v_variable
     * Set the name of the variable containing the latitudinal component of
     * IVT.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, ivt_v_variable)
    ///@}

    /** @name region_mask_variable
     * Set the name of the variable containing the mask defining the region of
     * interest.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, region_mask_variable)
    ///@}

    /** @name regional_moisture_flux_variable
     * Set the name of the array to store the result in. the default is
     * "regional_moisture_flux"
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, moisture_flux_variable)
    ///@}


protected:
    teca_regional_moisture_flux();
/*
    // helpers to get the variable names from either the incoming
    // request or the class member variable.
    std::string get_ivt_u_variable(const teca_metadata &request);
    std::string get_ivt_v_variable(const teca_metadata &request);
    std::string get_component_2_variable(const teca_metadata &request);
    std::string get_regional_moisture_flux_variable(const teca_metadata &request);
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
    std::string ivt_u_variable;
    std::string ivt_v_variable;
    std::string region_mask_variable;
    std::string moisture_flux_variable;
    teca_metadata moisture_flux_attributes;
};

#endif

