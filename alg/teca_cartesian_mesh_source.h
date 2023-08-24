#ifndef teca_cartesian_mesh_source_h
#define teca_cartesian_mesh_source_h

#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <functional>
#include <map>
#include <utility>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cartesian_mesh_source)

/** The signature of the callback used to specify user defined fields.
 * f(x, y, z, t) -> w
 * Given spatial coordinate axes x,y,z and the time t, return the
 * 3D field w. The first argument specifies what device the data should
 * be placed on.
 */
using field_generator_callback = std::function<p_teca_variant_array(int,
    const const_p_teca_variant_array &, const const_p_teca_variant_array &,
    const const_p_teca_variant_array &, double)>;

/** An object that bundles field name, the metadata attributes needed for I/O,
 * and a field generator callback. Use this with append_field_generator
 */
struct TECA_EXPORT field_generator
{
    std::string name;
    teca_metadata attributes;
    field_generator_callback generator;
};

using field_generator_t = field_generator;

inline
bool operator==(const field_generator &l, const field_generator &r)
{
    return l.name == r.name;
}

inline
bool operator!=(const field_generator &l, const field_generator &r)
{
    return l.name != r.name;
}

/** @brief
 * An algorithm that generates a teca_cartesian_mesh of the requested
 * spatial and temporal dimensions with optional user defined fields.
 *
 * @details
 * User defined fields are specified by passing callbacks and metadata
 * via field_generator and append_field_generator
 *
 * The spatial and temporal dimensions are set by the combination of
 *  whole_extent and  bounds.
 *
 * The names of coordinate axes are set by the combination
 * of  x_axis_variable,  y_axis_variable,  z_axis_variable,
 * and  t_axis_variable
 *
 * The units of the coordinate axes are set by the combination of
 *  x_axis_units,  y_axis_units,  z_axis_units,  calendar,
 * and  time_units.
 */
class TECA_EXPORT teca_cartesian_mesh_source : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cartesian_mesh_source)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cartesian_mesh_source)
    TECA_ALGORITHM_CLASS_NAME(teca_cartesian_mesh_source)
    ~teca_cartesian_mesh_source();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name coordinate_type_code
     * set/get the type code for generated coordinates. The default is a 64 bit
     * floating point type. Use teca_variant_array_code<NT>::get() to get
     * specific type codes for C++ POD types NT.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(unsigned int, coordinate_type_code)
    ///@}

    /** @name field_type_code
     * set/get the type code for generated fields. The default is a 64 bit
     * floating point type. Use teca_variant_array_code<NT>::get() to get
     * specific type codes for C++ POD types NT.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(unsigned int, field_type_code)
    ///@}

    /** @name whole_extent
     * set/get the global index space extent of the data.  the extents are
     * given by 8 values, 6 spatial plus 2 temporal, in the following order
     * [i0 i1 j0 j1 k0 k1 q0 q1] This should be the same on all ranks
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(unsigned long, whole_extent)

    /** Set the spatial extents from a metadata object following the
     * conventions defined by the  teca_cf_reader. If three_d is true the
     * extents in the z-direction are copied, otherwise they are set to 0.
     * Returns zero if successful and non-zero if the supplied metadata is
     * missing any of the requisite information.
     **/
    int set_spatial_extents(const teca_metadata &md, bool three_d = true);
    ///@}

    /** @name bounds
     * set/get the global bounds of the data. the bounds are 8 values 6 spatial
     * plus 2 temporal in the following order. [x0 x1 y0 y1 z0 z1 t0 t1]
     * this should be the same on all ranks.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(double, bound)

    /** Set the spatial bounds from a metadata object following the conventions
     * defined by the  teca_cf_reader. Returns zero if successful and
     * non-zero if the supplied metadata is missing any of the requisite
     * information.
     */
    int set_spatial_bounds(const teca_metadata &md, bool three_d = true);

    ///@}

    /** @name x_axis_variable
     * set the name of the variable to use for the coordinate axes and
     * optionally associated attributes.
     */
    ///@{
    /** set the name of the t_axis_variable */
    void set_x_axis_variable(const std::string &name);

    /** Set the name of the variable and its attributes. See
     * teca_array_attributes for more information.
     */
    void set_x_axis_variable(const std::string &name, const teca_metadata &atts);

    /** Set the name of the variable and its attributes using conventions
     * defined by the  teca_cf_reader. Returns zero if successful and
     * non-zero if the supplied metadata is missing any of the requisite
     * information.
     */
    int set_x_axis_variable(const teca_metadata &md);
    ///@}

    /** @name y_axis_variable
     * set the name of the variable to use for the coordinate axes and
     * optionally associated attributes.
     */
    ///@{
    /** set the name of the y_axis_variable */
    void set_y_axis_variable(const std::string &name);

    /** Set the name of the variable and its attributes. See
     * teca_array_attributes for more information.
     */
    void set_y_axis_variable(const std::string &name, const teca_metadata &atts);

    /** Set the name of the variable and its attributes using conventions
     * defined by the  teca_cf_reader. Returns zero if successful and
     * non-zero if the supplied metadata is missing any of the requisite
     * information.
     */
    int set_y_axis_variable(const teca_metadata &md);
    ///@}

    /** @name z_axis_variable
     * set the name of the variable to use for the coordinate axes and
     * optionally associated attributes.
     */
    ///@{
    /** set the name of the z_axis_variable */
    void set_z_axis_variable(const std::string &name);

    /** Set the name of the variable and its attributes. See
     * teca_array_attributes for more information.
     */
    void set_z_axis_variable(const std::string &name, const teca_metadata &atts);

    /** Set the name of the variable and its attributes using conventions
     * defined by the  teca_cf_reader. Returns zero if successful and
     * non-zero if the supplied metadata is missing any of the requisite
     * information.
     */
    int set_z_axis_variable(const teca_metadata &md);
    ///@}

    /** @name t_axis_variable
     * set the name of the variable to use for the coordinate axes and
     * optionally associated attributes.
     */
    ///@{
    /** set the name of the t_axis_variable */
    void set_t_axis_variable(const std::string &name);

    /** Set the calendar, and time units of the t_axis_variable */
    void set_calendar(const std::string &calendar, const std::string &units);

    /** Set the name of the variable and its attributes. See
     * teca_array_attributes for more information.
     */
    void set_t_axis_variable(const std::string &name,
        const teca_metadata &atts);

    /** Set the name of the variable and its attributes using conventions
     * defined by the  teca_cf_reader. Returns zero if successful and
     * non-zero if the supplied metadata is missing any of the requisite
     * information.
     */
    int set_t_axis_variable(const teca_metadata &md);

    /** Set the time axis using coordinate conventions defined by the
     * teca_cf_reader. When a time axis is provided values are served up from
     * the array rather than being generated. Execution control keys are also
     * made use of if present. Returns zero if successful and non-zero if the
     * supplied metadata is missing any of the requisite information.
     */
    int set_t_axis(const teca_metadata &md);

    /** Set the time axis directly.  When a time axis is provided values are
     * served up from the array rather than being generated. Execution control
     * keys are also made use of if present.
     */
    void set_t_axis(const p_teca_variant_array &t);
    ///@}

    /** @name output_metadata
     * Set the output metadata directly. The provided metadata must contain
     * "coordinates" as defined by the  teca_cf_reader because these are
     * required for mesh generation. Pipeline execution control keys as defined
     * by  teca_index_executive are also required. Calendaring metadata is
     * recommended. A copy of the passed object is made but "variables" are
     * replaced with those generated by this class, if any. As a result be sure
     * to specifiy field generators before calling this method. Returns 0 if
     * successful, and non-zero if the supplied metadata doesn't contain the
     * expected information. No error messages are sent to the terminal.
     */
    ///@{
    int set_output_metadata(const teca_metadata &md);
    ///@}

    /** @name append_field_generator
     * set a callback function f(x,y,z,t) that generates a field named name
     * x,y,z are coordinate axes in variant arrays, t is the double precision
     * time value.
     */
    ///@{
    void append_field_generator(const std::string &name,
        const teca_metadata &atts, field_generator_callback &callback);
    ///@}

    /** @name field_generator
     * Set/get the named callbacks that generate fields on the mesh. These
     * should be packaged in the field_generator struct so that field name
     * and attributes for I/O are provided together with the callback.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(field_generator_t, field_generator)
    ///@}

protected:
    teca_cartesian_mesh_source();

private:
    using teca_algorithm::get_output_metadata;

    /// implements the report phase of pipeline execution
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    /// implements the execute phase of pipeline execution
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    /// updates the modification state
    void set_modified() override;

    /// clears cached metadata in response to modification of algorithm properties
    void clear_cached_metadata();

private:
    unsigned int coordinate_type_code;
    unsigned int field_type_code;
    std::string x_axis_variable;
    std::string y_axis_variable;
    std::string z_axis_variable;
    std::string t_axis_variable;
    teca_metadata x_axis_attributes;
    teca_metadata y_axis_attributes;
    teca_metadata z_axis_attributes;
    teca_metadata t_axis_attributes;
    std::vector<unsigned long> whole_extents;
    std::vector<double> bounds;

    std::vector<field_generator_t> field_generators;

    struct internals_t;
    internals_t *internals;
};

#endif
