#ifndef teca_wrf_reader_h
#define teca_wrf_reader_h

#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_wrf_reader)

class teca_wrf_reader_internals;
using p_teca_wrf_reader_internals = std::shared_ptr<teca_wrf_reader_internals>;

/// A reader for data stored in WRF ARW format.
/**
 * Reads a set of arrays from single time step into an Arakawa C grid
 * mesh. The mesh is optionally subset.
 *
 * ### metadata keys:
 *
 *  | key                   | description |
 *  | ----                  | ----------- |
 *  | variables             | a list of all available variables |
 *  | [var]                 | a metadata object holding all NetCDF attributes for the |
 *  |                       | variable named [var] |
 *  | time variables        | a list of all variables with time as the only |
 *  |                       | dimension |
 *  | coordinates           | a metadata object holding names and arrays of the |
 *  |                       | coordinate axes |
 *  |  x_axis_variable      | name of x axis variable |
 *  |  y_axis_variable      | name of y axis variable |
 *  |  z_axis_variable      | name of z axis variable |
 *  |  t_axis_variable      | name of t axis variable |
 *  |  x                    | array of x coordinates |
 *  |  y                    | array of y coordinates |
 *  |  z                    | array of z coordinates |
 *  |  t                    | array of t coordinates |
 *  |  files                | list of files in this dataset |
 *  |  step_count           | list of the number of steps in each file |
 *  |  number_of_time_steps | total number of time steps in all files |
 *  |  whole_extent         | index space extent describing (nodal) dimensions of the |
 *  |                       | mesh |
 *
 * ### request keys:
 *
 *  | key        | description |
 *  | ----       | ----------- |
 *  | time_step  | the time step to read
 *  | arrays     | list of arrays to read
 *  | extent     | index space extents describing the subset of data to read
 *
 * ### output:
 *
 * generates a 2 or 3D teca_arakawa_c_grid mesh for the requested timestep on
 * the requested extent with the requested point based arrays and value at this
 * timestep for all time variables.
 */
class TECA_EXPORT teca_wrf_reader : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_wrf_reader)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_wrf_reader)
    TECA_ALGORITHM_CLASS_NAME(teca_wrf_reader)
    ~teca_wrf_reader();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // list of file names to open. if this is set the files_regex
    // is ignored.
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, file_name)

    // describe the set of files comprising the dataset. This
    // should contain the full path and regex describing the
    // file name pattern
    TECA_ALGORITHM_PROPERTY(std::string, files_regex)

    // the directory where metadata should be cached. if this is not specified
    // metadata is cached either with the data, in the CWD, or in the user's
    // home dir, which ever location succeeds first.
    TECA_ALGORITHM_PROPERTY(std::string, metadata_cache_dir)

    // set if the dataset has periodic boundary conditions
    TECA_ALGORITHM_PROPERTY(int, periodic_in_x)
    TECA_ALGORITHM_PROPERTY(int, periodic_in_y)
    TECA_ALGORITHM_PROPERTY(int, periodic_in_z)

    // time calendar and time unit if the user wants to
    // specify them
    TECA_ALGORITHM_PROPERTY(std::string, calendar)
    TECA_ALGORITHM_PROPERTY(std::string, t_units)

    // a way to infer time from the filename if the time axis is not
    // stored in the file itself. strftime format codes are used.
    // For example for the files:
    //
    //      my_file_20170516_00.nc
    //      my_file_20170516_03.nc
    //      ...
    //
    // the template would be
    //
    //      my_file_%Y%m%d_%H.nc
    TECA_ALGORITHM_PROPERTY(std::string, filename_time_template)

    // time values to use instead if time variable doesn't
    // exist.
    TECA_ALGORITHM_VECTOR_PROPERTY(double, t_value)

    // set/get the number of threads in the pool. setting
    // to less than 1 results in 1 - the number of cores.
    // the default is 1.
    TECA_ALGORITHM_PROPERTY(int, thread_pool_size)

protected:
    teca_wrf_reader();
    void clear_cached_metadata();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    virtual void set_modified() override;

private:
    std::vector<std::string> file_names;
    std::string files_regex;
    std::string metadata_cache_dir;
    std::string m_x_axis_variable;
    std::string m_y_axis_variable;
    std::string u_x_axis_variable;
    std::string u_y_axis_variable;
    std::string v_x_axis_variable;
    std::string v_y_axis_variable;
    std::string m_z_axis_variable;
    std::string w_z_axis_variable;
    std::string t_axis_variable;
    std::string calendar;
    std::string t_units;
    std::string filename_time_template;
    std::vector<double> t_values;
    int periodic_in_x;
    int periodic_in_y;
    int periodic_in_z;
    int thread_pool_size;
    p_teca_wrf_reader_internals internals;
};

#endif
