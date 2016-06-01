#include "teca_table_calendar.h"

#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif
#if defined(TECA_HAS_UDUNITS)
#include "calcalcs.h"
#endif
#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using std::string;
using std::vector;
using std::set;
using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_table_calendar::teca_table_calendar() :
    units(), calendar(), time_column("time"), year_column("year"),
    month_column("month"), day_column("day"), hour_column("hour"),
    minute_column("minute"), second_column(), output_column_prefix()
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_table_calendar::~teca_table_calendar()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_table_calendar::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_table_calendar":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, units,
            "CF-2 base date and units. eg \"days since January 1 1971\"")
        TECA_POPTS_GET(std::string, prefix, calendar,
            "CF-2 calendar type. one of: Gregorian, Julian, no_leap, or 360_day")
        TECA_POPTS_GET(std::string, prefix, time_column,
            "name of the column containing CF-2 time variable")
        TECA_POPTS_GET(std::string, prefix, year_column,
            "name of the column to store the computed year ")
        TECA_POPTS_GET(std::string, prefix, month_column,
            "name of the column to store the computed month ")
        TECA_POPTS_GET(std::string, prefix, day_column,
            "name of the column to store the computed day ")
        TECA_POPTS_GET(std::string, prefix, hour_column,
            "name of the column to store the computed hours ")
        TECA_POPTS_GET(std::string, prefix, minute_column,
            "name of the column to store the computed minutes ")
        TECA_POPTS_GET(std::string, prefix, second_column,
            "name of the column to store the computed seconds ")
        TECA_POPTS_GET(std::string, prefix, output_column_prefix,
            "prepended to all output column names")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_table_calendar::set_properties(
    const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, units)
    TECA_POPTS_SET(opts, std::string, prefix, calendar)
    TECA_POPTS_SET(opts, std::string, prefix, time_column)
    TECA_POPTS_SET(opts, std::string, prefix, year_column)
    TECA_POPTS_SET(opts, std::string, prefix, month_column)
    TECA_POPTS_SET(opts, std::string, prefix, day_column)
    TECA_POPTS_SET(opts, std::string, prefix, hour_column)
    TECA_POPTS_SET(opts, std::string, prefix, minute_column)
    TECA_POPTS_SET(opts, std::string, prefix, second_column)
    TECA_POPTS_SET(opts, std::string, prefix, output_column_prefix)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_table_calendar::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_table_calendar::execute" << endl;
#endif
    (void)port;
    (void)request;
#if !defined(TECA_HAS_UDUNITS)
    (void)input_data;
    TECA_ERROR("Calendaring features are not present")
    return nullptr;
#else
    // get the input table
    const_p_teca_table in_table
        = std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    // only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    if (!in_table)
    {
        if (rank == 0)
        {
            TECA_ERROR("Input is empty or not a table")
        }
        return nullptr;
    }

    // get calendar and unit system
    std::string units = this->units;
    if (units.empty() &&
        ((in_table->get_time_units(units)) && units.empty()))
    {
        TECA_ERROR("Units are missing")
        return nullptr;
    }

    std::string calendar = this->calendar;
    if (calendar.empty() &&
        ((in_table->get_calendar(calendar)) && calendar.empty()))
    {
        TECA_ERROR("Calendar is missing")
        return nullptr;
    }

    // values to convert are in the time column, grab that
    const_p_teca_variant_array time = in_table->get_column(this->time_column);
    if (!time)
    {
        TECA_ERROR("column \"" << this->time_column
            << "\" is not in the table")
        return nullptr;
    }

    // allocate the output table, add the requested columns
    p_teca_table out_table = teca_table::New();
    out_table->copy_metadata(in_table);

    unsigned long n_rows = time->size();

    p_teca_variant_array_impl<int> year;
    if (!this->year_column.empty())
    {
        std::string year_col = this->output_column_prefix + this->year_column;
        out_table->declare_columns(year_col, int());
        year = std::static_pointer_cast
            <teca_variant_array_impl<int>>(out_table->get_column(year_col));
        year->reserve(n_rows);
    }

    p_teca_variant_array_impl<int> month;
    if (!this->month_column.empty())
    {
        std::string month_col = this->output_column_prefix + this->month_column;
        out_table->declare_columns(month_col, int());
        month = std::static_pointer_cast
            <teca_variant_array_impl<int>>(out_table->get_column(month_col));
        month->reserve(n_rows);
    }

    p_teca_variant_array_impl<int> day;
    if (!this->day_column.empty())
    {
        std::string day_col = this->output_column_prefix + this->day_column;
        out_table->declare_columns(day_col, int());
        day = std::static_pointer_cast
            <teca_variant_array_impl<int>>(out_table->get_column(day_col));
        day->reserve(n_rows);
    }

    p_teca_variant_array_impl<int> hour;
    if (!this->hour_column.empty())
    {
        std::string hour_col = this->output_column_prefix + this->hour_column;
        out_table->declare_columns(hour_col, int());
        hour = std::static_pointer_cast
            <teca_variant_array_impl<int>>(out_table->get_column(hour_col));
        hour->reserve(n_rows);
    }

    p_teca_variant_array_impl<int> minute;
    if (!this->minute_column.empty())
    {
        std::string minute_col = this->output_column_prefix + this->minute_column;
        out_table->declare_columns(minute_col, int());
        minute = std::static_pointer_cast
            <teca_variant_array_impl<int>>(out_table->get_column(minute_col));
        minute->reserve(n_rows);
    }

    p_teca_variant_array_impl<double> second;
    if (!this->second_column.empty())
    {
        std::string second_col = this->output_column_prefix + this->second_column;
        out_table->declare_columns(second_col, double());
        second = std::static_pointer_cast
            <teca_variant_array_impl<double>>(out_table->get_column(second_col));
        second->reserve(n_rows);
    }

    // make the date computations
    TEMPLATE_DISPATCH(
        const teca_variant_array_impl,
        time.get(),

        const NT *curr_time = static_cast<TT*>(time.get())->get();

        for (unsigned long i = 0; i < n_rows; ++i)
        {
            int curr_year = 0;
            int curr_month = 0;
            int curr_day = 0;
            int curr_hour = 0;
            int curr_minute = 0;
            double curr_second = 0;

            if (calcalcs::date(curr_time[i], &curr_year, &curr_month,
                &curr_day, &curr_hour, &curr_minute, &curr_second,
                units.c_str(), calendar.c_str()))
            {
                TECA_ERROR("Failed to compute the date at row " << i)
                return nullptr;
            }

            if (year)
                out_table << curr_year;

            if (month)
                out_table << curr_month;

            if (day)
                out_table << curr_day;

            if (hour)
                out_table << curr_hour;

            if (minute)
                out_table << curr_minute;

            if (second)
                out_table << curr_second;
        }
        )

    // add the rest of the table
    out_table->concatenate_cols(in_table);

    return out_table;
#endif
}
