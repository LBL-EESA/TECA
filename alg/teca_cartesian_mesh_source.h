#ifndef teca_cartesian_mesh_source_h
#define teca_cartesian_mesh_source_h

#include "teca_algorithm.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cartesian_mesh_source)

/**
An algorithm that constructs and serves up a Cartesian mesh
of the specified dimensions.
*/
class teca_cartesian_mesh_source : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cartesian_mesh_source)
    ~teca_cartesian_mesh_source();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set/get the type code for generated coordinates.
    // default is 32 bit floating point.
    TECA_ALGORITHM_PROPERTY(int, coordinate_type_code)

    // set/get the global index space extent of the data.
    // this should be the same on all processing elements.
    TECA_ALGORITHM_VECTOR_PROPERTY(unsigned long, whole_extent)

    // set/get the global bounds of the data.
    // this should be the same on all processing elements.
    TECA_ALGORITHM_VECTOR_PROPERTY(double, bound)

    // set the variable to use for the coordinate axes.
    // the defaults are: x => lon, y => lat, z = plev,
    // t => time
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, t_axis_variable)

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
    std::string x_axis_variable;
    std::string y_axis_variable;
    std::string z_axis_variable;
    std::string t_axis_variable;
    std::vector<unsigned long> whole_extents;
    std::vector<double> bounds;

    struct internals_t;
    internals_t *internals;
};

#endif
