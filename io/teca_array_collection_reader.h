#ifndef teca_array_collection_reader_h
#define teca_array_collection_reader_h

#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"
#include "teca_array_collection.h"

#include <vector>
#include <string>
#include <mutex>


TECA_SHARED_OBJECT_FORWARD_DECL(teca_array_collection_reader)

/// A reader for collections of arrays stored in NetCDF format.
/**
 * The reader reads requested arrays into a teca_array_collection.
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
 *  | files                 | the list of files in this dataset |
 *  | step_count            | a list containing the number of steps in each file indiexed by file |
 *  | index_initializer_key | set to the string "number_of_time_steps" |
 *  | number_of_time_steps  | set to the total number of time steps in all files |
 *  | index_request_key     | set to the string "temporal_extent" |
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
 *  | x_axis_variable | the name of x axis variable |
 *  | y_axis_variable | the name of y axis variable |
 *  | z_axis_variable | the name of z axis variable |
 *  | t_axis_variable | the name of t axis variable |
 *  | x               | the array of x coordinates |
 *  | y               | the array of y coordinates |
 *  | z               | the array of z coordinates |
 *  | t               | the array of t coordinates |
 *
 * ### request keys:
 *
 *  | key             | description |
 *  | ----            | ----------- |
 *  | temporal_extent | holds an inclusive range of time step to read [i0, i1] |
 *  | arrays          | holds a list of arrays to read |
 *
 * ### output:
 * The reader generates a 1,2 or 3D cartesian mesh for the requested timestep
 * on the requested extent with the requested point based arrays and value at
 * this timestep for all time variables.
 */
class TECA_EXPORT teca_array_collection_reader : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_array_collection_reader)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_array_collection_reader)
    TECA_ALGORITHM_CLASS_NAME(teca_array_collection_reader)
    ~teca_array_collection_reader();

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

protected:
    teca_array_collection_reader();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void set_modified() override;
    void clear_cached_metadata();

private:
    std::vector<std::string> file_names;
    std::string files_regex;
    std::string t_axis_variable;
    std::string calendar;
    std::string t_units;
    std::string filename_time_template;
    std::vector<double> t_values;
    int max_metadata_ranks;

    struct teca_array_collection_reader_internals;
    teca_array_collection_reader_internals *internals;
};

#endif
