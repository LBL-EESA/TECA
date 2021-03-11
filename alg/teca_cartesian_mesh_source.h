#ifndef teca_cartesian_mesh_source_h
#define teca_cartesian_mesh_source_h

#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <functional>
#include <map>
#include <utility>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cartesian_mesh_source)

/** The signature of the callback used to specify user defined fields.
 * f(x, y, z, t) -> w
 * Given spatial coordinate axes x,y,z and the time t, return the
 * 3D field w.
 */
using field_generator_callback = std::function<p_teca_variant_array(
    const const_p_teca_variant_array &, const const_p_teca_variant_array &,
    const const_p_teca_variant_array &, double)>;

/** An object that bundles field name, the metadata attributes needed for I/O,
 * and a field generator callback. Use this with ::append_field_generator
 */
struct field_generator
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
 * via @ref field_generator and @ref append_field_generator
 *
 * The spatial and temporal dimensions are set by the combination of
 * @ref whole_extent and @ref bounds.
 *
 * The names of coordinate axes are set by the combination
 * of @ref x_axis_variable, @ref y_axis_variable, @ref z_axis_variable,
 * and @ref t_axis_variable
 *
 * The units of the coordinate axes are set by the combination of
 * @ref x_axis_units, @ref y_axis_units, @ref z_axis_units, @ref calendar,
 * and @ref time_units.
 */
class teca_cartesian_mesh_source : public teca_algorithm
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

    /** @anchor coordinate_type_code
     * @name coordinate_type_code
     * set/get the type code for generated coordinates. The default is a 64 bit
     * floating point type. Use teca_variant_array_code<NT>::get() to get
     * specific type codes for C++ POD types NT.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(unsigned int, coordinate_type_code)
    ///@}

    /** @anchor field_type_code
     * @name field_type_code
     * set/get the type code for generated fields. The default is a 64 bit
     * floating point type. Use teca_variant_array_code<NT>::get() to get
     * specific type codes for C++ POD types NT.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(unsigned int, field_type_code)
    ///@}

    /** @anchor whole_extent
     * @name whole_extent
     * set/get the global index space extent of the data.  the extents are
     * given by 8 values, 6 spatial plus 2 temporal, in the following order
     * [i0 i1 j0 j1 k0 k1 q0 q1] This should be the same on all ranks
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(unsigned long, whole_extent)
    ///@}

    /** @anchor bounds
     * @name bounds
     * set/get the global bounds of the data. the bounds are 8 values 6 spatial
     * plus 2 temporal in the following order. [x0 x1 y0 y1 z0 z1 t0 t1]
     * this should be the same on all ranks.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(double, bound)

    /** Set the spatial bounds from a metadata object following the conventions
     * defined by the teca_cf_reader. This provides an easy way to get valid
     * mesh bounds from an existing dataset where the producer of the dataset
     * has followed those conventions. Returns zero if successful and non-zero
     * if the supplied metadata is missing any of the requisite information.
     */
    int set_spatial_bounds(const teca_metadata &md);
    ///@}

    /** @anchor x_axis_variable
     * @name x_axis_variable
     * set the variable to use for the coordinate axes. the default is: lon
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_variable)
    ///@}

    /** @anchor y_axis_variable
     * @name y_axis_variable
     * set the variable to use for the coordinate axes. the defaults is: lat
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_variable)
    ///@}

    /** @anchor z_axis_variable
     * @name z_axis_variable
     * set the variable to use for the coordinate axes. the default is: plev
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_variable)
    ///@}

    /** @anchor t_axis_variable
     * @name t_axis_variable
     * set the variable to use for the coordinate axes.  * the default is: time
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, t_axis_variable)
    ///@}

    /** @anchor x_axis_units
     * @name x_axis_units
     * set the units of spatial axes. The defaults is: degrees_east
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_units)
    ///@}

    /** @anchor y_axis_units
     * @name y_axis_units
     * set the units of spatial axes. The defaults is: degrees_north
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_units)
    ///@}

    /** @anchor z_axis_units
     * @name z_axis_units
     * set the units of spatial axes. The defaults is: Pa
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_units)
    ///@}

    /** @anchor calendar
     * @name calendar
     * Set/get the calendar. The default is "Gregorian".
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, calendar)

    /** Set the time units and calendar from a metadata object following the
     * conventions defined by the teca_cf_reader. This provides an easy way to
     * get calendaring information from an existing dataset where the producer
     * of the dataset has followed those conventions. Returns zero if
     * successful and non-zero if the supplied metadata is missing any of the
     * requisite information.
     */
    int set_calendar(const teca_metadata &md);
    ///@}

    /** @anchor time_units
     * @name time_units
     * Set/get the calendar. The default is "seconds since 1970-01-01 00:00:00".
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, time_units)
    ///@}

    /** @anchor append_field_generator
     * @name append_field_generator
     * set a callback function f(x,y,z,t) that generates a field named name
     * x,y,z are coordinate axes in variant arrays, t is the double precision
     * time value.
     */
    ///@{
    void append_field_generator(const std::string &name,
        const teca_metadata &atts, field_generator_callback &callback);
    ///@}

    /** @anchor field_generator
     * @name field_generator
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
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void set_modified() override;
    void clear_cached_metadata();

private:
    unsigned int coordinate_type_code;
    unsigned int field_type_code;
    std::string x_axis_variable;
    std::string y_axis_variable;
    std::string z_axis_variable;
    std::string t_axis_variable;
    std::string x_axis_units;
    std::string y_axis_units;
    std::string z_axis_units;
    std::string calendar;
    std::string time_units;
    std::vector<unsigned long> whole_extents;
    std::vector<double> bounds;

    std::vector<field_generator_t> field_generators;

    struct internals_t;
    internals_t *internals;
};

#endif
