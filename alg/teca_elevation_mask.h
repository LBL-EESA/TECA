#ifndef teca_elevation_mask_h
#define teca_elevation_mask_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_elevation_mask)

/** @brief
 * Generates a mask indicating where mesh points with a vertical pressure
 * coordinate lie above the surface of the Earth.  The mask is set to 1 where
 * data is above the Earth's surface and 0 otherwise.
 *
 * @details
 * Given a 3D height field containing the altitude of each point in meters
 * above mean sea level, and a 2D height field corresponding to height in
 * meters above mean sea level of the surface of the Earth, generate a mask
 * that is 1 where the 3D point is on or above the surface of the Earth and 0
 * where it is below.
 *
 * The name of the 3D height field is specified by the  mesh_height_variable
 * property. The name of the 2D height field containing elevation of the
 * Earth's surface is specified by the  surface_elevation_variable property.
 *
 * The 3D mesh height field must be provided on input 0, and the 2D surface
 * height field on input 1. Use the mask_names property to name the output
 * mask.  If more than one name is provided each name will reference a pointer
 * to the mask. Consider using names of the form X_valid in which case the
 * output is compatible with the teca_valid_value_mask and will be treated
 * as missing values by down stream algorithms.
 *
 * If the simulation does not provide the 3D height field, for simulations
 * where the acceleration due to the Earth's gravity is assumed constant,
 * teca_geopotential_height can generate the 3D height field.
 *
 * The primary use case of this algorithm is when dealing with calculations on
 * 3D meshes with a vertical pressure coordinate and there is a need to
 * identify and treat specially the mesh points that are below the surface of
 * the Earth. There are a number of alternatives available depending on the
 * data.  If your data has a _FillValue where data is below the surface then
 * use teca_valid_value_mask instead of this algorithm. If your data has
 * surface pressure field use teca_pressure_level_mask instead of this
 * algorithm. If your dataset has surface temperature, and mean sea level
 * pressure fields then use teca_surface_pressure to generate the surface
 * pressure field and use teca_pressure_level_mask instead of this algorithm.
 */
class TECA_EXPORT teca_elevation_mask : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_elevation_mask)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_elevation_mask)
    TECA_ALGORITHM_CLASS_NAME(teca_elevation_mask)
    ~teca_elevation_mask();

    /** @name program_options
     * report/initialize to/from Boost program options objects.
     */
    ///@{
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()
    ///@}

    /** @name mesh_height_variable
     * Set the name of the 3D height field
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, mesh_height_variable)
    ///@}

    /** @name surface_elevation_variable
     * Set the name of the variable containing the elevation of the Earth's
     * surface.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, surface_elevation_variable)
    ///@}

    /** @name mask_variables
     * set the names of the variables to store the generated mask in
     * each variable will contain a reference to the mask
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, mask_variable)
    ///@}

protected:
    teca_elevation_mask();

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
    std::string mesh_height_variable;
    std::string surface_elevation_variable;
    std::vector<std::string> mask_variables;
    struct internals_t;
};

#endif
