#ifndef teca_cartesian_mesh_regrid_h
#define teca_cartesian_mesh_regrid_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_variant_array.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cartesian_mesh_regrid)

/** @brief
 * Transfers data between spatially overlapping meshes of potentially different
 * resolutions.
 *
 * @details
 * an algorithm that transfers data between cartesian meshes defined in the
 * same world coordinate system but potentially different resolutions. nearest
 * or linear interpolation are supported.
 *
 * By default the first input is the target mesh. the second input is the
 * source mesh. This can be changed by setting the target_input property.
 *
 * the arrays to move from source to target can be selected using add_array api
 * or in the request key "arrays". this is a spatial regriding operation for
 * temporal regriding see teca_mesh_temporal_regrid.
 */
class TECA_EXPORT teca_cartesian_mesh_regrid : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cartesian_mesh_regrid)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cartesian_mesh_regrid)
    TECA_ALGORITHM_CLASS_NAME(teca_cartesian_mesh_regrid)
    ~teca_cartesian_mesh_regrid();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name array
     * set the list of arrays to move from the source to the target
     */
    //@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, array)
    //@}

    /** @name target_input
     * set the input connection which provides the output geometry.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, target_input)
    ///@}

    /** @name interpolation_mode
     * set the interpolation mode used in transfering data between meshes of
     * differing resolution.  in nearest mode value at the nearest grid point
     * is used, in linear mode bi/tri linear interpolation is used.
     */
    //@{
    enum {nearest=0, linear=1};
    TECA_ALGORITHM_PROPERTY(int, interpolation_mode)
    void set_interpolation_mode_nearest(){ interpolation_mode = nearest; }
    void set_interpolation_mode_linear(){ interpolation_mode = linear; }
    //@}



protected:
    teca_cartesian_mesh_regrid();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::vector<std::string> arrays;
    int target_input;
    int interpolation_mode;
};

#endif
