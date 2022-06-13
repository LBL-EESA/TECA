#ifndef teca_cartesian_mesh_coordinate_transform_h
#define teca_cartesian_mesh_coordinate_transform_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_variant_array.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cartesian_mesh_coordinate_transform)

/** @breif Transforms the coordinate axes of a stretched Cartesian mesh without
 * modifying the fields on the mesh.
 *
 * @details
 * Transform the spatial coorindate axes of an input dataset such that a given
 * target bounding box is covered by the output mesh.  The relative spacing of
 * the input coordinates are maintained.  The target bounding box must be
 * provided as a 6-tuple in the form [x0, x1, y0, y1, z0, z1] where 3
 * consecutive 2-tuples define the bounds in each coordinate direction. If any
 * of the 2-tuples are set such that the low bound is greater than the high
 * bound the transform is skipped in that direction and existing coordinate
 * axis array is passed through.  The transform only modifies the coodinate
 * axis arrays, not the fields defined on the mesh.  The coordinate axes may
 * optionally be renamed and new units provided.
 */
class TECA_EXPORT teca_cartesian_mesh_coordinate_transform : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cartesian_mesh_coordinate_transform)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cartesian_mesh_coordinate_transform)
    TECA_ALGORITHM_CLASS_NAME(teca_cartesian_mesh_coordinate_transform)
    ~teca_cartesian_mesh_coordinate_transform();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name target_bounds
     * 6 double precision values that define the output coordinate axis
     * bounds, specified in the following order : [x0 x1 y0 y1 z0 z1].
     * The Cartesian mesh is transformed such that its coordinatres span
     * the specified target bounds while maintaining relative spacing of
     * original input coordinate points. Pass [1, 0] for each axis that
     * should not be transformed.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(double, target_bound)

    /** set the target bounds from a metadata object following conventions
     * defined by the teca_cf_reader. returns 0 if sccessful or nonzero if
     * the metadata object is missing the requisite keys.
     */
    int set_target_bounds(const teca_metadata &md);
    ///@}

    /** @name x_axis_variable
     * Set the name of the variable to use for the x coordinate axis. When set
     * to an empty string (the default) the existing name is passed through.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_variable)
    ///@}

    /** @name y_axis_variable
     * Set the name of the variable to use for the y coordinate axis. When set
     * to an empty string (the default) the existing name is passed through.
     * Set the name of the variable to use for the y coordinate axis.  An empty
     * string disables the renaming of this dimension passing through the
     * original name.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_variable)
    ///@}
    /** @name z_axis_variable
     * Set the name of the variable to use for the z coordinate axis. When set
     * to an empty string (the default) the existing name is passed through.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_variable)
    ///@}

    /** @name x_axis_units
     * Set the units for the x coordinate axis. When set to an empty string
     * (the default) existing uints if any are passed through unmodified.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_units)
    ///@}

    /** @name y_axis_units
     * Set the units for the y coordinate axis. When set to an empty string
     * (the default) existing uints if any are passed through unmodified.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_units)
    ///@}

    /** @name z_axis_units
     * Set the units for the z coordinate axis. When set to an empty string
     * (the default) existing uints if any are passed through unmodified.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_units)
    ///@}

protected:
    teca_cartesian_mesh_coordinate_transform();

    void set_modified() override;

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
    std::vector<double> target_bounds;
    std::string x_axis_variable;
    std::string y_axis_variable;
    std::string z_axis_variable;
    std::string x_axis_units;
    std::string y_axis_units;
    std::string z_axis_units;

    struct internals_t;
    internals_t *internals;
};

#endif
