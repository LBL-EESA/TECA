#ifndef teca_surface_pressure_estimate_h
#define teca_surface_pressure_estimate_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_surface_pressure_estimate)

/** @breif
 * An algorithm that computes an estimate surface pressure from mean sea level
 * pressure, surface temperature, and surface elevation and the dry adiabatic
 * lapse rate.
 */
class teca_surface_pressure_estimate : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_surface_pressure_estimate)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_surface_pressure_estimate)
    TECA_ALGORITHM_CLASS_NAME(teca_surface_pressure_estimate)
    ~teca_surface_pressure_estimate();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name surface_temperature_variable
     * The name of the array that contains the surface temperature field.
     * The default is "tas"
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, surface_temperature_variable)
    ///@}

    /** @name sea_level_pressure_variable
     * The name of the array that contains the mean sea level pressure field.
     * The default is "psl"
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, sea_level_pressure_variable)
    ///@}

    /** @name surface_elevation_variable
     * The name of the array that contains the surface elevation field.
     * The default is "z"
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, surface_elevation_variable)
    ///@}


    /** @name surface_pressure_variable
     * set the name of the array to store the computed surface pressure in.
     * the default is "ps"
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, surface_pressure_variable)
    ///@}

protected:
    teca_surface_pressure_estimate();

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
    std::string surface_temperature_variable;
    std::string sea_level_pressure_variable;
    std::string surface_elevation_variable;
    std::string surface_pressure_variable;
};

#endif
