#include "teca_event_filter.h"

#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_distance_function.h"
#include "teca_geometry.h"

#include <iostream>
#include <string>
#include <set>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif
#if defined(TECA_HAS_UDUNITS)
#include "calcalcs.h"
#endif
#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using std::cerr;
using std::endl;

// --------------------------------------------------------------------------
teca_event_filter::teca_event_filter() :
    time_column(""), step_column(""), x_coordinate_column(""),
    y_coordinate_column(""), start_time(std::numeric_limits<double>::lowest()),
    end_time(std::numeric_limits<double>::max()), step_interval(1)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_event_filter::~teca_event_filter()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_event_filter::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_event_filter":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, time_column,
            "name of the column containing time axis. default "
            "\"\" disbales the filter")
        TECA_POPTS_GET(std::string, prefix, step_column,
            "name of the column containing time steps. default "
            "\"\" disables the filter")
        TECA_POPTS_GET(std::string, prefix, x_coordinate_column,
            "name of the column containing x cooridnates. default "
            "\"\" disables the filter")
        TECA_POPTS_GET(std::string, prefix, y_coordinate_column,
            "name of the column containing y cooridnates. default "
            "\"\" disables the filter")
        TECA_POPTS_GET(double, prefix, start_time,
            "include all events after the start time. default is -infinty")
        TECA_POPTS_GET(double, prefix, end_time,
            "include all events before the end time. default is +inifinity")
        TECA_POPTS_GET(long, prefix, step_interval, "output in time at this interval")
        TECA_POPTS_MULTI_GET(std::vector<unsigned long>,
            prefix, region_sizes, "the number of points in each region")
        TECA_POPTS_MULTI_GET(std::vector<double>,
            prefix, region_x_coordinates, "list of x coordinates describing the regions")
        TECA_POPTS_MULTI_GET(std::vector<double>,
            prefix, region_y_coordinates, "list of y coordinates describing the regions")
        TECA_POPTS_MULTI_GET(std::vector<int>,
            prefix, region_ids, "list of numeric ids identifying each region. "
            " if not provided sequential ids are generated")
        TECA_POPTS_MULTI_GET(std::vector<std::string>,
            prefix, region_names, "list of names identifying each region. "
            "if not provided names are generated from ids")
        TECA_POPTS_MULTI_GET(std::vector<std::string>,
            prefix, region_names, "list of long/readable names identifying "
            "each region. if not provided names are generated from ids")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_event_filter::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, time_column)
    TECA_POPTS_SET(opts, std::string, prefix, x_coordinate_column)
    TECA_POPTS_SET(opts, std::string, prefix, y_coordinate_column)
    TECA_POPTS_SET(opts, double, prefix, start_time)
    TECA_POPTS_SET(opts, double, prefix, end_time)
    TECA_POPTS_SET(opts, std::vector<unsigned long>, prefix, region_sizes)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, region_x_coordinates)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, region_y_coordinates)
    TECA_POPTS_SET(opts, std::vector<int>, prefix, region_ids)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, region_names)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, region_long_names)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_event_filter::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_event_filter::execute" << endl;
#endif
    (void)port;
    (void)request;

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
    unsigned long n_rows = in_table->get_number_of_rows();
    unsigned long n_cols = in_table->get_number_of_columns();

    // construct the output
    p_teca_table out_table = teca_table::New();
    out_table->copy_metadata(in_table);
    out_table->copy_structure(in_table);

    // filter along time axis
    const_p_teca_variant_array t;
    std::set<unsigned long> valid_time;
    if (!this->time_column.empty())
    {
        t = in_table->get_column(this->time_column);
        if (!t)
        {
            TECA_ERROR("time column \"" << this->time_column
                << "\" is not in the table")
            return nullptr;
        }
        TEMPLATE_DISPATCH(const teca_variant_array_impl,
            t.get(),
            const NT *pt = static_cast<const TT*>(t.get())->get();
            for (unsigned long i = 0; i < n_rows; ++i)
            {
                if ((pt[i] >= this->start_time) && (pt[i] <= this->end_time))
                    valid_time.insert(i);
            }
            )
        if (valid_time.empty())
        {
            TECA_WARNING("No rows satisfy the time filter")
        }
    }

    // apply the interval filter
    const_p_teca_variant_array step;
    std::set<unsigned long> valid_step;
    if (!this->step_column.empty())
    {
        step = in_table->get_column(this->step_column);
        if (!step)
        {
            TECA_ERROR("step column \"" << this->step_column
                << "\" is not in the table")
            return nullptr;
        }
        TEMPLATE_DISPATCH_I(const teca_variant_array_impl,
            step.get(),
            const NT *pstep = static_cast<const TT*>(step.get())->get();
            for (unsigned long i = 0; i < n_rows; ++i)
            {
                if (!(pstep[i] % this->step_interval))
                    valid_step.insert(i);
            }
            )
        if (valid_step.empty())
        {
            TECA_WARNING("No rows satisfy the step interval filter")
        }
    }

    // get the number of regions to filter by
    unsigned long n_regions = 0;

    std::set<unsigned long> valid_pos;
    std::map<unsigned long, int> valid_pos_ids;

    if (!this->x_coordinate_column.empty())
    {
        n_regions = this->region_sizes.size();
        if (!n_regions)
        {
            TECA_ERROR("no regions to filter by specified")
            return nullptr;
        }

        // grab the event coorinates
        const_p_teca_variant_array x =
            in_table->get_column(this->x_coordinate_column);
        if (!x)
        {
            TECA_ERROR("x coordinate column \"" << this->x_coordinate_column
                << "\" is not in the table")
            return nullptr;
        }

        const_p_teca_variant_array y =
            in_table->get_column(this->y_coordinate_column);
        if (!y)
        {
            TECA_ERROR("y coordinate column \"" << this->y_coordinate_column
                << "\" is not in the table")
            return nullptr;
        }

        // create internal data structures
        std::vector<unsigned long> rstarts(this->region_starts);
        if (rstarts.empty())
        {
            // generate starts
            rstarts.reserve(n_regions);
            rstarts.push_back(0);
            for (size_t  i = 0; i < n_regions; ++i)
                rstarts.push_back(rstarts[i] + this->region_sizes[i]);
        }

        std::vector<int> rids(this->region_ids);
        if (rids.empty())
        {
            // generate ids
            rids.reserve(n_regions);
            for (size_t  i = 0; i < n_regions; ++i)
                rids.push_back(i);
        }

        std::vector<std::string> rnames(this->region_names);
        if (rnames.empty())
        {
            // generate names
            std::ostringstream oss;
            rnames.reserve(n_regions);
            for (size_t  i = 0; i < n_regions; ++i)
            {
                oss.str("");
                oss << "region_" << i;
                rnames.push_back(oss.str());
            }
        }

        // allocate space for name and id
        TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
            x.get(),

            const NT *px = static_cast<const TT*>(x.get())->get();
            const NT *py = static_cast<const TT*>(y.get())->get();

            for (unsigned long i = 0; i < n_rows; ++i)
            {
                double ptx = px[i];
                double pty = py[i];
                for (unsigned long j = 0; j < n_regions; ++j)
                {
                    double *polyx = this->region_x_coordinates.data() + rstarts[j];
                    double *polyy = this->region_y_coordinates.data() + rstarts[j];
                    if (teca_geometry::point_in_poly(ptx, pty, polyx, polyy, this->region_sizes[j]))
                    {
                        valid_pos.insert(i);
                        break;
                    }
                }
            }
            )
            if (valid_pos.empty())
            {
                TECA_WARNING("No rows satisfy the spatial filter")
            }
    }

    // identify rows that meet all criteria
    std::vector<unsigned long> valid_rows;
    for (unsigned long i = 0; i < n_rows; ++i)
    {
        bool have_time = valid_time.count(i);
        bool have_step = valid_step.count(i);
        bool have_pos = valid_pos.count(i);
        if ((t && step && n_regions && have_time && have_step && have_pos) ||
            (t && !step && n_regions && have_time && have_pos) ||
            (t && step && !n_regions && have_time && have_step) ||
            (t && !step && !n_regions && have_time) ||
            (!t && step && n_regions && have_step && have_pos) ||
            (!t && !step && n_regions && have_pos) ||
            (!t && step && !n_regions && have_step))
            valid_rows.push_back(i);
    }

    // for each column copy the valid rows
    unsigned long n_valid = valid_rows.size();
    for (unsigned long j = 0; j < n_cols; ++j)
    {
        const_p_teca_variant_array in_col = in_table->get_column(j);

        p_teca_variant_array out_col = out_table->get_column(j);
        out_col->resize(n_valid);

        TEMPLATE_DISPATCH(teca_variant_array_impl,
            out_col.get(),
            const NT *pin = static_cast<const TT*>(in_col.get())->get();
            NT *pout = static_cast<TT*>(out_col.get())->get();
            for (unsigned long i = 0; i < n_valid; ++i)
            {
                pout[i] = pin[valid_rows[i]];
            }
            )
    }

    return out_table;
}
