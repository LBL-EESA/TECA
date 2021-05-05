#ifndef teca_multi_cf_reader_h
#define teca_multi_cf_reader_h

#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"
#include "teca_cf_reader.h"

#include <set>
#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_multi_cf_reader)

class teca_multi_cf_reader_internals;
using p_teca_multi_cf_reader_internals = std::shared_ptr<teca_multi_cf_reader_internals>;

/// A reader for data stored in NetCDF CF format in multiple files.
/**
 * The data read is presented to the down stream as a single dataset
 *
 * use the add_reader method to specify regular expression and corresponding
 * list of variables to read. a reader, not necessarily the same one, must be
 * selected to provide the time and spatial axes.
 *
 * this reader could handle spatio-temporal interpolations as well, however
 * that is currently not implemented. as a result all data is expected to be
 * on the same coordinate system.
 *
 * A number of algorithm properties modify run time behavior, most of these
 * are exposed from teca_cf_reader. see the teca_cf_reader for details.
 *
 * The reader may be initialized via a configuration file. The configuration
 * file consists of name = value pairs and flags organized in sections.
 * Sections are declared using []. There is an optional  global section
 * followed by a number of [cf_reader] sections. Each [cf_reader] section
 * consists of a name(optional), a regex, a list of variables, a provides_time
 * flag(optional) and a provides geometry flag(optional). At least one section
 * must contain a provides_time and provides geometry flag. The global section
 * may contain a data_root. Occurrences of the string %data_root% in the regex
 * are replaced with the value of data_root.
 *
 * The following example configures the reader to read hus,ua and va.
 *
 * ```
 * # TECA multi_cf_reader config
 *
 * data_root = /opt/TECA_data/HighResMIP/ECMWF-IFS-HR-SST-present
 *
 * [cf_reader]
 * regex = %data_root%/hus/hus.*\.nc$
 * variables = hus
 * provides_time
 * provides_geometry
 *
 * [cf_reader]
 * regex = %data_root%/va/va.*\.nc$
 * variables = va
 *
 * [cf_reader]
 * regex = %data_root%/ua/ua.*\.nc$
 * variables = ua
 * ```
 */
class teca_multi_cf_reader : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_multi_cf_reader)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_multi_cf_reader)
    TECA_ALGORITHM_CLASS_NAME(teca_multi_cf_reader)
    ~teca_multi_cf_reader();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /**
     * Set the MCF configuration file that describes the dataset to read.
     * Each section in the MCF file adds an internal reader.
     */
    int set_input_file(const std::string &input_file);
    std::string get_input_file() { return this->input_file; }

    /**
     * Adds a reader to the collection and at the same time specifies how it
     * will be used. This is alternative way to configure the multi_cf_reader
     * instead of providing the configuration via an MCF file (see
     * set_input_file).
     */
    int add_reader(const std::string &regex,
        const std::string &key, int provides_time,
        int provides_geometry,
        const std::vector<std::string> &variables);

    /// sets the reader that provides the time axis
    int set_time_reader(const std::string &key);

    /// sets the reader that provides the mesh geometry
    int set_geometry_reader(const std::string &key);

    /// adds to the list of variables that a reader will provide
    int add_variable_reader(const std::string &key,
        const std::string &variable);

    /// sets the list of variable that a reader will provide.
    int set_variable_reader(const std::string &key,
        const std::vector<std::string> &variable);

    /// get the list of variables that the reader will serve up
    void get_variables(std::vector<std::string> &vars);

    /** @name periodic_in_x
     * Set to indicate the presence of a periodic boundary in the x direction.
     * If set this will override the corresponding setting from the MCF file
     * for all internal readers.
     */
    ///@{
    void set_periodic_in_x(int flag);
    int get_periodic_in_x() const;
    ///@}

    /** @name x_axis_variable
     * Set the variable to use for the mesh x-axis. If set this will override
     * the corresponding setting from the MCF file for all internal readers.
     */
    ///@{
    void set_x_axis_variable(const std::string &var);
    std::string get_x_axis_variable() const;
    ///@}

    /** @name y_axis_variable
     * Set the variable to use for the mesh y-axis. If set this will override
     * the corresponding setting from the MCF file for all internal readers.
     */
    ///@{
    void set_y_axis_variable(const std::string &var);
    std::string get_y_axis_variable() const;
    ///@}

    /** @name z_axis_variable
     * Set the variable to use for the mesh z-axis. Leaving the z-axis empty
     * results in a 2D mesh. You must set this to the correct vertical
     * coordinate dimension to produce a 3D mesh. If set this will override
     * the corresponding setting from the MCF file for all internal readers.
     */
    ///@{
    void set_z_axis_variable(const std::string &var);
    std::string get_z_axis_variable() const;
    ///@}

    /** @name t_axis_variable_
     * Set the variable to use for the mesh t-axis. Default "time". Setting
     * this to an empty string disables the time axis. If set this will
     * override the corresponding setting from the MCF file for all internal
     * readers.
     */
    ///@{
    void set_t_axis_variable(const std::string &var);
    std::string get_t_axis_variable() const;
    ///@}

    /** @name calendar
     * Use this to override the calendar, or set one when specifying t_values
     * directly. If set this will override the corresponding setting from the
     * MCF file for all internal readers.
     */
    ///@{
    void set_calendar(const std::string &calendar);
    std::string get_calendar() const;
    ///@}

    /** @name t_units
     * Use this to set or override the time units. This is necessary when
     * specifying time values directly. If set this will override the
     * corresponding setting from the MCF file for all internal readers.
     */
    ///@{
    void set_t_units(const std::string &units);
    std::string get_t_units() const;
    ///@}

    /** @name filename_time_template
     * a way to infer time from the filename if the time axis is not stored in
     * the file itself. If set this will override the corresponding setting
     * from the MCF file for all internal readers.
     *
     * strftime format codes are used. For example for the files:
     * ```
     *      my_file_20170516_00.nc
     *      my_file_20170516_03.nc
     *      ...
     * ```
     * the template would be
     * ```
     *      my_file_%Y%m%d_%H.nc
     * ```
     */
    ///@{
    void set_filename_time_template(const std::string &templ);
    std::string get_filename_time_template() const;
    ///@}

    /** @name t_values
     * Set the time values to use instead if a time variable doesn't exist or
     * you need to override it. If set this will override the corresponding
     * setting from the MCF file for all internal readers.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(double, t_value)
    ///@}

    /** @name max_metadata_ranks
     * set/get the number of ranks used to read the time axis. If set this
     * will override the corresponding setting from the MCF file for all
     * internal readers.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, max_metadata_ranks)
    ///@}

    /** @name periodic_in_x
     * Set to indicate the presence of a periodic boundary in the x direction.
     * If set this will override the corresponding setting from the MCF file
     * for all internal readers.
     */
    ///@{
    ///@}

    /** @name clamp_dimensions_of_one
     * If set the requested extent will be clamped in a given direction if the
     * coorinate axis in that direction has a length of 1 and the requested
     * extent would be out of bounds. This is a work around to enable loading
     * 2D data with a vertical dimension of 1, into a 3D mesh and should be
     * used with caution.
     */
    ///@{
    void set_clamp_dimensions_of_one(int flag);
    int get_clamp_dimensions_of_one() const;
    ///@}

protected:
    teca_multi_cf_reader();

private:
    void clear_cached_metadata();

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void set_modified() override;

private:
    std::string input_file;
    std::string x_axis_variable;
    std::string y_axis_variable;
    std::string z_axis_variable;
    std::string t_axis_variable;
    std::string calendar;
    std::string t_units;
    std::string filename_time_template;
    std::vector<double> t_values;
    int periodic_in_x;
    int max_metadata_ranks;
    int clamp_dimensions_of_one;

    p_teca_multi_cf_reader_internals internals;
};

#endif
