#ifndef teca_table_calendar_h
#define teca_table_calendar_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_calendar)

/// an algorithm that transforms NetCDF CF-2 time
/// variable into an absolute date.
/**
Transform NetCDF CF-2 time variable into an absolute
date. By default the "time" column is used, but this
can be over road by set_active_column methods. the
table must have temporal metadata containing base date
and calendar units following the CF-2 convention.
the output table will contain year,month,day,hours,
minutes columns.

NOTE: this should be used in serial, as the udunits
package loads an xml file in each instance. The
CalCalcs package also has thread safety issues.
*/
class teca_table_calendar : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_calendar)
    ~teca_table_calendar();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name of the column to convert. If this is
    // not set "time" is used.
    TECA_ALGORITHM_PROPERTY(std::string, time_column)

    // set the CF-2 base date and units. If this is not
    // set then the units are obtained from table metadata.
    TECA_ALGORITHM_PROPERTY(std::string, units)

    // set the CF-2 calendar. if this is not set then
    // the calendar is obtained from the table metadata.
    TECA_ALGORITHM_PROPERTY(std::string, calendar)

    // set the names of the output columns. The defaults are
    // "year", "month", "day", "hour", "minute", "second"
    // setting any of these to the empty string "", will
    // suppress their output.
    TECA_ALGORITHM_PROPERTY(std::string, year_column)
    TECA_ALGORITHM_PROPERTY(std::string, month_column)
    TECA_ALGORITHM_PROPERTY(std::string, day_column)
    TECA_ALGORITHM_PROPERTY(std::string, hour_column)
    TECA_ALGORITHM_PROPERTY(std::string, minute_column)
    TECA_ALGORITHM_PROPERTY(std::string, second_column)

    // set output prefix. the prefix (optional) is prepended
    // to the column names.
    TECA_ALGORITHM_PROPERTY(std::string, output_column_prefix)

protected:
    teca_table_calendar();

private:
    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string units;
    std::string calendar;
    std::string time_column;
    std::string year_column;
    std::string month_column;
    std::string day_column;
    std::string hour_column;
    std::string minute_column;
    std::string second_column;
    std::string output_column_prefix;
};

#endif
