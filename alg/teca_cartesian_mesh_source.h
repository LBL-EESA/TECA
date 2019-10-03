#ifndef teca_cartesian_mesh_source_h
#define teca_cartesian_mesh_source_h

#include "teca_algorithm.h"
#include <functional>
#include <map>
#include <utility>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cartesian_mesh_source)

// f(x, y, z, t)
// given coordinat axes x,y,z return the field
using field_generator_callback = std::function<p_teca_variant_array(
    const const_p_teca_variant_array &, const const_p_teca_variant_array &,
    const const_p_teca_variant_array &, double)>;

struct field_generator
{
    std::string name;
    field_generator_callback generator;
};

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

using field_generator_t = field_generator;

/**
An algorithm that constructs and serves up a Cartesian mesh
of the specified dimensions.
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

    // set/get the type code for generated coordinates.
    // default is 32 bit floating point.
    TECA_ALGORITHM_PROPERTY(int, coordinate_type_code)
    TECA_ALGORITHM_PROPERTY(int, field_type_code)

    // set/get the global index space extent of the data.  the extents are
    // given by 8 values, 6 spatial plus 2 temporal, in the following order
    // [i0 i1 j0 j1 k0 k1 q0 q1]
    // this should be the same on all ranks elements.
    TECA_ALGORITHM_VECTOR_PROPERTY(unsigned long, whole_extent)

    // set/get the global bounds of the data. the bounds are 8 values 6 spatial
    // plus 2 temporal in the following order.
    // [x0 x1 y0 y1 z0 z1 t0 t1]
    // this should be the same on all ranks elements.
    TECA_ALGORITHM_VECTOR_PROPERTY(double, bound)

    // set the variable to use for the coordinate axes.
    // the defaults are: x => lon, y => lat, z = plev,
    // t => time
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, t_axis_variable)

    // set the units of spatial axes. The defaults are:
    // degrees_east, degrees_north, and pressure_level
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_units)
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_units)
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_units)

    // number of time steps to generate
    TECA_ALGORITHM_PROPERTY(std::string, calendar)
    TECA_ALGORITHM_PROPERTY(std::string, time_units)

    // set the named callbacks to generate fields on the mesh
    // A callback f must have the signature f(x,y,z,t).
    TECA_ALGORITHM_VECTOR_PROPERTY(field_generator_t, field_generator);

    // set a callback function f(x,y,z,t) that generates a field named name
    // x,y,z are coordinate axes in variant arrays, t is the double precision
    // time value.
    void append_field_generator(const std::string &name,
        field_generator_callback &callback);

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
    int coordinate_type_code;
    int field_type_code;
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
