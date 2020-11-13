#ifndef teca_multi_cf_reader_h
#define teca_multi_cf_reader_h

#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"

#include <set>
#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_multi_cf_reader)

class teca_multi_cf_reader_internals;
using p_teca_multi_cf_reader_internals = std::shared_ptr<teca_multi_cf_reader_internals>;

/// a reader for data stored in NetCDF CF format in multiple files
/**
a reader for data stored in NetCDF CF format in multiple files.  the data read
is presented to the down stream as a single dataset

use the add_reader method to specify regular experiession and corresponding
list of variables to read. a reader, not necessarily the same one, must be
selected to provide the time and spatial axes.

this reader could handle spatio-temporal interpolations as well, however that
is currently not implemented. as a result all data is expected to be on the
same coordinate system.

A number of algorithm properties modify run time behavior, most of these are
exposed from teca_cf_reader. see the teca_cf_reader for details.

The reader may be initialized via a configuration file. The configuration file
consists of name = value pairs and flags organized in sections. Sections are
declared using []. There is an optional  global section followed by a number of
[cf_reader] sections. Each [cf_reader] section consists of a name(optional), a
regex, a list of variables, a provides_time flag(optional) and a provides
geometry flag(optional). At least one section must contain a provides_time and
provides geometry flag. The global section may contain a data_root. Occurances
of the string %data_root% in the regex are replaced with the value of
data_root.

The following example configures the reader to read hus,ua and va.

```
# TECA multi_cf_reader config

data_root = /opt/TECA_data/HighResMIP/ECMWF-IFS-HR-SST-present

[cf_reader]
regex = %data_root%/hus/hus.*\.nc$
variables = hus
provides_time
provides_geometry

[cf_reader]
regex = %data_root%/va/va.*\.nc$
variables = va

[cf_reader]
regex = %data_root%/ua/ua.*\.nc$
variables = ua
```
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

    // set the file name describing the dataset to read
    int set_input_file(const std::string &input_file);

    // adds a reader to the collection and at the same time specifies
    // how it will be used.
    int add_reader(const std::string &key,
        const std::string &files_regex,
        int provides_time, int provides_geometry,
        const std::vector<std::string> &variables);

    // sets the reader that provides the time axis
    int set_time_reader(const std::string &key);

    // sets the reader that provides the mesh geometry
    int set_geometry_reader(const std::string &key);

    // adds to the list of variables that a reader will provide
    int add_variable_reader(const std::string &key,
        const std::string &variable);

    // sets the list of variable that a reader will provide.
    int set_variable_reader(const std::string &key,
        const std::vector<std::string> &variable);

    // get the list of variables that the reader will serve up
    void get_variables(std::vector<std::string> &vars);

    // set if the dataset has periodic boundary conditions
    TECA_ALGORITHM_PROPERTY(int, periodic_in_x)
    TECA_ALGORITHM_PROPERTY(int, periodic_in_y)
    TECA_ALGORITHM_PROPERTY(int, periodic_in_z)

    // set the variable to use for the coordinate axes.
    // the defaults are: x => lon, y => lat, z = "",
    // t => "time". leaving z empty will result in a 2D
    // mesh.
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, t_axis_variable)

    // time calendar and time unit if the user wants to
    // specify them
    TECA_ALGORITHM_PROPERTY(std::string, t_calendar)
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

    // set/get the number of ranks used to read the time axis.
    // the default value of 1024 ranks works well on NERSC
    // Cori scratch file system and may not be optimal on
    // other systems.
    TECA_ALGORITHM_PROPERTY(int, max_metadata_ranks)

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
    std::string x_axis_variable;
    std::string y_axis_variable;
    std::string z_axis_variable;
    std::string t_axis_variable;
    std::string t_calendar;
    std::string t_units;
    std::string filename_time_template;
    std::vector<double> t_values;
    int periodic_in_x;
    int periodic_in_y;
    int periodic_in_z;
    int max_metadata_ranks;

    p_teca_multi_cf_reader_internals internals;
};

#endif
