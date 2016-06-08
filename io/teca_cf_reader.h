#ifndef teca_cf_reader_h
#define teca_cf_reader_h

#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"

#include <vector>
#include <string>
#include <mutex>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cf_reader)

class teca_cf_reader_internals;
using p_teca_cf_reader_internals = std::shared_ptr<teca_cf_reader_internals>;

/// a reader for data stored in NetCDF CF format
/**
a reader for data stored in NetCDF CF format

reads a set of arrays from  single time step into a cartesian
mesh. the mesh is optionally subset.

metadata keys:
    variables - a list of all available variables.
    <var> -  a metadata object holding all NetCDF attributes for the variable named <var>
    time variables - a list of all variables with time as the only dimension
    coordinates - a metadata object holding names and arrays of the coordinate axes
        x_variable - name of x axis variable
        y_variable - name of y axis variable
        z_variable - name of z axis variable
        t_variable - name of t axis variable
        x - array of x coordinates
        y - array of y coordinates
        z - array of z coordinates
        t - array of t coordinates
    files - list of files in this dataset
    step_count - list of the number of steps in each file
    number_of_time_steps - total number of time steps in all files
    whole_extent - index space extent describing (nodal) dimensions of the mesh

request keys:
    time_step - the time step to read
    arrays - list of arrays to read
    extent - index space extents describing the subset of data to read

output:
    generates a 1,2 or 3D cartesian mesh for the requested timestep
    on the requested extent with the requested point based arrays
    and value at this timestep for all time variables.
*/
class teca_cf_reader : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cf_reader)
    ~teca_cf_reader();

    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cf_reader)

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // describe the set of files comprising the dataset. This
    // should contain the full path and regex describing the
    // file name pattern
    TECA_ALGORITHM_PROPERTY(std::string, files_regex)

    // a file name to open. if this is set the files_regex
    // is ignored.
    TECA_ALGORITHM_PROPERTY(std::string, file_name)

    // set the variable to use for the coordinate axes.
    // the defaults are: x => lon, y => lat, z = "",
    // t => "time". leaving z empty will result in a 2D
    // mesh.
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, t_axis_variable)

    // set/get the number of threads in the pool. setting
    // to less than 1 results in 1 - the number of cores.
    // the default is 1.
    TECA_ALGORITHM_PROPERTY(int, thread_pool_size)

protected:
    teca_cf_reader();
    void clear_cached_metadata();

private:
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    virtual void set_modified() override;

private:
    std::string files_regex;
    std::string file_name;
    std::string x_axis_variable;
    std::string y_axis_variable;
    std::string z_axis_variable;
    std::string t_axis_variable;
    int thread_pool_size;
    p_teca_cf_reader_internals internals;
};

#endif
