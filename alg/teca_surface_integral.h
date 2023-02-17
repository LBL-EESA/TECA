#ifndef teca_surface_integral_h
#define teca_surface_integral_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_array_attributes.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_surface_integral)

/// Compute the integral over a geographic region on the surface of the Earth.
/**  Compute the integral over a geographic region on the surface of the Earth.
 * This is esepcially useful for calculating net flux normal to the surface
 * over the specified region and may be useful for integrating other variables
 * which have units of something over square meters.
 *
 * The surface inetgral of a variable \f$F\f$ over a region enclosed by curve
 * \f$C\f$ is given by:
 *
 * \f[
 * F_{net} = -r_e^2 \iint\limits_{C} F \; sin \phi \; d \theta d \phi
 * \f]
 *
 * where:
 *
 *  F  :  a 2D field in units of something per square meter <br>
 *  \f$C\f$  :  A curve enclosing a region on the surface of the Earth  <br>
 *  \f$\phi\f$  :  \f$(90 -  \f$degrees latitude\f$) \frac{\pi}{180}\f$ radians <br>
 *  \f$\theta\f$  :  \f$(\f$degrees longitude\f$) \frac{\pi}{180}\f$  radians <br>
 *
 * Note: the minus sign out front is because we flip the limits of integration
 * when going from latitude to \f$phi\f$ coordinates.
 *
 * Requirements:
 * 1. A list of surface variables
 * 2. A list of names to use when packaging the results
 * 3. A 2D mask where non-zero entries define the geographic region of
 *    interest. ::teca_shape_file_mask is an algorithm that can be used
 *    to generate such a mask from an ESRI shapefile.
 *
 * Produces a table with one column for each calculation made and
 * columns for time step and time.
 *
 * @attention This algorithm does not currently handle missing values.
 */
class TECA_EXPORT teca_surface_integral : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_surface_integral)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_surface_integral)
    TECA_ALGORITHM_CLASS_NAME(teca_surface_integral)
    ~teca_surface_integral();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name input_variables
     * Set the name of the variables to integrate.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, input_variable)
    ///@}

    /** @name output_variables
     * Set the name of variables storing the results.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, output_variable)
    ///@}

    /** @name region_mask_variable
     * Set the name of the variable containing the mask defining the region of
     * interest.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, region_mask_variable)
    ///@}

    /** @name output_prefix
     * Set a string to prepend to the output variable name
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, output_prefix)
    ///@}

protected:
    teca_surface_integral();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
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
    std::vector<std::string> input_variables;
    std::vector<std::string> output_variables;
    std::vector<teca_array_attributes> output_attributes;
    std::string region_mask_variable;
    std::string output_prefix;
};

#endif

