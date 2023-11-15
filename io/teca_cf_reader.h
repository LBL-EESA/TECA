#ifndef teca_cf_reader_h
#define teca_cf_reader_h

#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cf_reader)

class teca_cf_reader_internals;
using p_teca_cf_reader_internals = std::shared_ptr<teca_cf_reader_internals>;

/// A reader for Cartesian mesh based data stored in NetCDF CF format.
/**
 * Reads a set of arrays from  single time step into a teca_cartesian_mesh
 * dataset. The reader responds to requests for specific arrays and the data
 * may be optionally subset via extent and bounds request keys.
 *
 * The time varying dataset to read is identified by a regular expression
 * identifying a set of files. Note, regular expressions are similar and more
 * powerful than the more familiar shell glob but the control characters have
 * different meanings.
 *
 * ### metadata keys:
 *
 *  | key                   | description |
 *  | ----                  | ----------- |
 *  | variables             | a list of all available variables. |
 *  | attributes            | a metadata object holding all NetCDF attributes for the variables |
 *  | coordinates           | a metadata object holding names and arrays of the coordinate axes |
 *  | files                 | list of files in this dataset |
 *  | step_count            | list of the number of steps in each file |
 *  | index_initializer_key | number_of_time_steps |
 *  | number_of_time_steps  | total number of time steps in all files |
 *  | index_request_key     | time_step |
 *  | whole_extent          | index space extent describing (nodal) dimensions of the mesh |
 *  | bounds                | world coordinate space bounding box covered by the mesh |
 *
 * ### attribute metadata:
 *
 *  | key             | description |
 *  | ----            | ----------- |
 *  | [variable name] | a metadata object holding all NetCDF attributes, and |
 *  |                 | TECA specific per-array metadata |
 *
 * ### cooridnate metadata:
 *
 *  | key             | description |
 *  | ----            | ----------- |
 *  | x_axis_variable | name of x axis variable |
 *  | y_axis_variable | name of y axis variable |
 *  | z_axis_variable | name of z axis variable |
 *  | t_axis_variable | name of t axis variable |
 *  | x               | array of x coordinates |
 *  | y               | array of y coordinates |
 *  | z               | array of z coordinates |
 *  | t               | array of t coordinates |
 *
 * ### request keys:
 *
 *  | key       | description |
 *  | ----      | ----------- |
 *  | time_step | the time step to read |
 *  | arrays    | list of arrays to read |
 *  | extent    | index space extents describing the subset of data to read |
 *  | bounds    | world space bounds describing the subset of data to read |
 *
 * ### output:
 * The reader generates a 1,2 or 3D cartesian mesh for the requested timestep
 * on the requested extent with the requested point based arrays and value at
 * this timestep for all time variables.
 */
class TECA_EXPORT teca_cf_reader : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cf_reader)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cf_reader)
    TECA_ALGORITHM_CLASS_NAME(teca_cf_reader)
    ~teca_cf_reader();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @name file_name
     * Set a list of files to open. If this is used then the files_regex is
     * ignored.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, file_name)
    ///@}

    /** @name files_regex
     * Set a regular expression identifying the set of files comprising the
     * dataset. This should contain the full path to the files and the regular
     * expression.  Only the final component of a path may contain a regex.
     * Be aware that regular expression control characters do not have the
     * same meaning as shell glob control characters. When used in a shell
     * regular expression control characters need to be quoted or escaped to
     * prevent the shell from interpreting them.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, files_regex)
    ///@}

    /** @name periodic_in_x
     * A flag that indicates a periodic bondary in the x direction
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, periodic_in_x)
    ///@}

    /** @name periodic_in_y
     * A flag that indicates a periodic bondary in the y direction
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, periodic_in_y)
    ///@}

    /** @name periodic_in_z
     * A flag that indicates a periodic bondary in the z direction
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, periodic_in_z)
    ///@}

    /** @name x_axis_variable
     * Set the name of the variable to use for the x coordinate axis.
     * An empty string disables this dimension.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_variable)
    ///@}

    /** @name y_axis_variable
     * Set the name of the variable to use for the y coordinate axis.
     * An empty string disables this dimension.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_variable)
    ///@}
    /** @name z_axis_variable
     * Set the name of the variable to use for the z coordinate axis.
     * An empty string disables this dimension.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_variable)
    ///@}

    /** @name t_axis_variable
     * Set the name of the variable to use for the t coordinate axis.
     * An empty string disables this dimension.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, t_axis_variable)
    ///@}

    /** @name calendar
     * Override the calendar.  When specified the values takes precedence over
     * the values found in the file.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, calendar)
    ///@}

    /** @name t_units
     * Override the time units. When specified the value takes precedence over
     * the values found in the file.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, t_units)
    ///@}

    /** @name filename_time_template
     * a way to infer time from the filename if the time axis is not stored in
     * the file itself. std::get_time format codes are used.  If a calendar is
     * not specified then the standard calendar is used. If time units are not
     * specified then the time units will be "days since %Y-%m-%d 00:00:00"
     * where Y,m, and d are computed from the filename of the first file. set
     * t_axis_variable to an empty string to use.
     *
     * For example, for the list of files:
     *
     * > my_file_20170516_00.nc
     * > my_file_20170516_03.nc
     * > ...
     *
     * the template would be
     *
     * > my_file_%Y%m%d_%H.nc
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, filename_time_template)
    ///@}

    /** @name t_value
     * an explicit list of double precision time values to use.  set
     * t_axis_variable to an empty string to use.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(double, t_value)
    ///@}

    /** @name max_metadata_ranks
     * set/get the number of ranks used to read the time axis.  the default
     * value of 1024 ranks works well on NERSC Cori scratch file system and may
     * not be optimal on other systems.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, max_metadata_ranks)
    ///@}

    /** @name clamp_dimensions_of_one
     * If set the requested extent will be clamped in a given direction if the
     * coorinate axis in that dircetion has a length of 1 and the requested extent
     * would be out of bounds. This exists to deal with non-conformant data and
     * should be used with caution.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, clamp_dimensions_of_one)
    ///@}

    /** @name collective_buffer
     * Enables MPI I/O colective buffering. Collective buffering is only valid
     * when the spatial partitioner is enabled and the number of spatial
     * partitions is equal to the number of MPI ranks, and the code is single
     * threaded. This is an experimental feature.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, collective_buffer)
    ///@}

protected:
    teca_cf_reader();
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
    std::string x_axis_variable;
    std::string y_axis_variable;
    std::string z_axis_variable;
    std::string t_axis_variable;
    std::string calendar;
    std::string t_units;
    std::string filename_time_template;
    std::vector<double> t_values;
    int periodic_in_x;
    int periodic_in_y;
    int periodic_in_z;
    int max_metadata_ranks;
    int clamp_dimensions_of_one;
    int collective_buffer;
    p_teca_cf_reader_internals internals;
};

#endif
