#include "teca_tc_classify.h"

#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_distance_function.h"
#include "teca_geometry.h"

#include <iostream>
#include <string>
#include <sstream>

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

namespace internal
{
// data describing the default geographic regions
// storms are sorted by
int reg_ids[] = {0, 1, 2, 3, 4, 5, 6, 7, 7, 4};
unsigned long reg_sizes[] = {5, 5, 6, 6, 14, 14, 7, 5, 7, 5};
unsigned long reg_starts[] = {0, 5, 10, 16, 22, 36, 50, 57, 62, 69};

const char *reg_names[] = {"SI", "SWP",
    "NWP", "NI", "NA", "NEP", "SEP", "SA"};

const char *reg_long_names[] = {"S Indian", "SW Pacific", "NW Pacific",
    "N Indian", "N Atlantic", "NE Pacific", "SE Pacific", "S Atlantic"};

// since we want to allow for an arbitrary
// set of polys, we can't hard code up a search
// optimization structure. but we can order them
// from most likeley+smallest to least likely+largest
double reg_lon[] = {
    // 0 S Indian (green)
    136, 136, 20, 20, 136,
    // 1 SW Pacific (pink)
    136, 216, 216, 136, 136,
    // 2 NW Pacific (orange)
    104, 180, 180, 98.75, 98.75, 104,
    // 3 N Indian (purple)
    20, 104, 98.75, 98.75, 20, 20,
    // 4 N Atlantic (red_360)
    282, 284, 284, 278, 268, 263, 237, 237, 224, 200, 200, 360.1, 360.1, 282,
    // 5 NE Pacific (yellow)
    200, 180, 180, 282, 284, 284, 278, 268, 263, 237, 237, 224, 200, 200,
    // 6 SE Pacific (cyan)
    216, 216, 289, 289, 298, 298, 216,
    // 7 S Atlantic (blue_0)
    -0.1, 20, 20, -0.1, -0.1,
    // 7 S Atlantic (blue_360)
    298, 298, 289, 289, 360.1, 360.1, 298,
    // 4 N Atlantic (red_0)
    20, 20, -0.1, -0.1, 20};

double reg_lat[] = {
    // S Indian (green)
    0, -90, -90, 0, 0,
    // SW Pacific (pink)
    -90, -90, 0, 0, -90,
    // NW Pacific (orange)
    0, 0, 90, 90, 9, 0,
    // N Indian (purple)
    0, 0, 9, 90, 90, 0,
    // N Atlantic (red_360)
    0, 3, 8.5, 8.5, 17, 17, 43, 50, 62, 62, 90, 90, 0, 0,
    // NE Pacific (yellow)
    90, 90, 0, 0, 3, 8.5, 8.5, 17, 17, 43, 50, 62, 62, 90,
    // SE Pacific (cyan)
    0, -90, -90, -52, -19.5, 0, 0,
    // S Atlantic (blue_0)
    0, 0, -90, -90, 0,
    // S Atlantic (blue_360)
    0, -19.5, -52, -90, -90, 0, 0,
    // N Atlantic (red_0)
    90, 0, 0, 90, 90};

// Saffir-Simpson scale prescribes the following limits:
// CAT wind km/h
// -1:   0- 62  :  Tropical depression
//  0:  63-117  :  Tropical storm
//  1: 119-153
//  2: 154-177
//  3: 178-209
//  4: 210-249
//  5:    >250
template<typename n_t>
int classify_saphir_simpson(n_t w)
{
    // 1 m/s -> 3.6 Km/h
    n_t w_kmph = n_t(3.6)*w;
    if (w_kmph <= n_t(62.0))
        return -1;
    else
    if (w_kmph <= n_t(117.0))
        return 0;
    else
    if (w_kmph <= n_t(153.0))
        return 1;
    else
    if (w_kmph <= n_t(177.0))
        return 2;
    else
    if (w_kmph <= n_t(209.0))
        return 3;
    else
    if (w_kmph <= n_t(249.0))
        return 4;
    return 5;
}
};

// --------------------------------------------------------------------------
teca_tc_classify::teca_tc_classify() :
    track_id_column("track_id"), time_column("time"), x_coordinate_column("lon"),
    y_coordinate_column("lat"), surface_wind_column("surface_wind"),
    sea_level_pressure_column("sea_level_pressure")
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    // initialize the default regions
    size_t n_regs = sizeof(internal::reg_sizes)/sizeof(unsigned long);
    this->region_sizes.assign(internal::reg_sizes, internal::reg_sizes+n_regs);
    this->region_starts.assign(internal::reg_starts, internal::reg_starts+n_regs);
    this->region_ids.assign(internal::reg_ids, internal::reg_ids+n_regs);

    size_t n_pts = sizeof(internal::reg_lon)/sizeof(double);
    this->region_x_coordinates.assign(internal::reg_lon, internal::reg_lon+n_pts);
    this->region_y_coordinates.assign(internal::reg_lat, internal::reg_lat+n_pts);

    size_t n_names = sizeof(internal::reg_names)/sizeof(char*);
    this->region_names.assign(internal::reg_names, internal::reg_names+n_names);
    this->region_long_names.assign(internal::reg_long_names,internal::reg_long_names+n_names);
}

// --------------------------------------------------------------------------
teca_tc_classify::~teca_tc_classify()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_tc_classify::get_properties_description(
    const std::string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_tc_classify":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, track_id_column,
            "name of the column containing track ids")
        TECA_POPTS_GET(std::string, prefix, time_column,
            "name of the column containing time stamps")
        TECA_POPTS_GET(std::string, prefix, surface_wind_column,
            "name of column containing wind speeds")
        TECA_POPTS_GET(std::string, prefix, x_coordinate_column,
            "name of the column containing x cooridnates")
        TECA_POPTS_GET(std::string, prefix, y_coordinate_column,
            "name of the column containing y cooridnates")
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
            prefix, region_long_names, "list of long/readable names identifying "
            "each region. if not provided names are generated from ids")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_tc_classify::set_properties(
    const std::string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, track_id_column)
    TECA_POPTS_SET(opts, std::string, prefix, time_column)
    TECA_POPTS_SET(opts, std::string, prefix, surface_wind_column)
    TECA_POPTS_SET(opts, std::string, prefix, x_coordinate_column)
    TECA_POPTS_SET(opts, std::string, prefix, y_coordinate_column)
    TECA_POPTS_SET(opts, std::vector<unsigned long>, prefix, region_sizes)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, region_x_coordinates)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, region_y_coordinates)
    TECA_POPTS_SET(opts, std::vector<int>, prefix, region_ids)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, region_names)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, region_long_names)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_tc_classify::execute(
    unsigned int port, const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_tc_classify::execute" << endl;
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
    // get calendar and unit system
    std::string units;
    if ((in_table->get_time_units(units)) && units.empty())
    {
        TECA_ERROR("Units are missing")
        return nullptr;
    }

    std::string calendar;
    if ((in_table->get_calendar(calendar)) && calendar.empty())
    {
        TECA_ERROR("Calendar is missing")
        return nullptr;
    }

    // get the track ids
    const_p_teca_int_array track_ids =
        std::dynamic_pointer_cast<const teca_int_array>(
            in_table->get_column(this->track_id_column));

    if (!track_ids)
    {
        TECA_ERROR("column \"" << this->track_id_column
            << "\" is not in the table")
        return nullptr;
    }

    // the spatial coorinates
    const_p_teca_variant_array x =
        in_table->get_column(this->x_coordinate_column);

    if (!x)
    {
        TECA_ERROR("column \"" << this->x_coordinate_column
            << "\" is not in the table")
        return nullptr;
    }

    const_p_teca_variant_array y =
        in_table->get_column(this->y_coordinate_column);

    if (!y)
    {
        TECA_ERROR("column \"" << this->y_coordinate_column
            << "\" is not in the table")
        return nullptr;
    }

    // time axis
    const_p_teca_variant_array time =
        in_table->get_column(this->time_column);

    if (!time)
    {
        TECA_ERROR("column \"" << this->time_column
            << "\" is not in the table")
        return nullptr;
    }

    // get the surface wind speeds
    const_p_teca_variant_array surface_wind =
        in_table->get_column(this->surface_wind_column);

    if (!surface_wind)
    {
        TECA_ERROR("column \"" << this->surface_wind_column
            << "\" is not in the table")
        return nullptr;
    }

    // get the surface wind speeds
    const_p_teca_variant_array sea_level_pressure =
        in_table->get_column(this->sea_level_pressure_column);

    if (!sea_level_pressure)
    {
        TECA_ERROR("column \"" << this->sea_level_pressure_column
            << "\" is not in the table")
        return nullptr;
    }

    // scan the track ids and build the random access
    // data structure
    std::vector<unsigned long> track_starts(1, 0);
    size_t n_rows = track_ids->size();
    const int *pids = track_ids->get();
    for (size_t i = 1; i < n_rows; ++i)
        if (pids[i] != pids[i-1])
            track_starts.push_back(i);
    track_starts.push_back(n_rows);
    size_t n_tracks = track_starts.size() - 1;

    // record track id
    p_teca_long_array out_ids = teca_long_array::New(n_tracks);
    long *pout_ids = out_ids->get();
    for (size_t i =0; i < n_tracks; ++i)
        pout_ids[i] = pids[track_starts[i]];

    // record track start time
    p_teca_variant_array start_time = time->new_instance(n_tracks);
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        start_time.get(),
        const NT *ptime = static_cast<const TT*>(time.get())->get();
        NT *pstart_time = static_cast<TT*>(start_time.get())->get();
        for (size_t i = 0; i < n_tracks; ++i)
            pstart_time[i] = ptime[track_starts[i]];
        )

    // record track start position
    p_teca_variant_array start_x = x->new_instance(n_tracks);
    p_teca_variant_array start_y = x->new_instance(n_tracks);

    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        start_x.get(),

        const NT *px = static_cast<const TT*>(x.get())->get();
        const NT *py = static_cast<const TT*>(y.get())->get();

        NT *pstart_x = static_cast<TT*>(start_x.get())->get();
        NT *pstart_y = static_cast<TT*>(start_y.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = track_starts[i];

            pstart_x[i] = px[q];
            pstart_y[i] = py[q];
        }
        )

    // compute the storm duration
    p_teca_variant_array duration = time->new_instance(n_tracks);
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        duration.get(),
        const NT *ptime = static_cast<const TT*>(time.get())->get();
        NT *pduration = static_cast<TT*>(duration.get())->get();
        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long first = track_starts[i];
            unsigned long last = track_starts[i+1] - 1;
            pduration[i] = ptime[last] - ptime[first];
        }
        )

    // compute the distance traveled
    p_teca_variant_array length = x->new_instance(n_tracks);

    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        length.get(),

        const NT *px = static_cast<const TT*>(x.get())->get();
        const NT *py = static_cast<const TT*>(y.get())->get();

        NT *plength = static_cast<TT*>(length.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long track_start = track_starts[i];
            unsigned long npts = track_starts[i+1] - track_start - 1;

            plength[i] = NT();

            for (unsigned long j = 0; j < npts; ++j)
            {
                unsigned long p = track_start + j;
                unsigned long q = p + 1;

                plength[i] += teca_distance(px[p], py[p], px[q], py[q]);
            }
        }
        )

    // rank the track on Saphir-Simpson scale
    // record the max wind speed, and position of it
    p_teca_int_array category = teca_int_array::New(n_tracks);
    int *pcategory = category->get();

    p_teca_variant_array max_surface_wind
        = surface_wind->new_instance(n_tracks);

    p_teca_unsigned_long_array max_surface_wind_id
        = teca_unsigned_long_array::New(n_tracks);

    unsigned long *pmax_surface_wind_id
        = max_surface_wind_id->get();

    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        max_surface_wind.get(),

        const NT *psurface_wind =
            static_cast<const TT*>(surface_wind.get())->get();

        NT *pmax_surface_wind =
            static_cast<TT*>(max_surface_wind.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long track_start = track_starts[i];
            unsigned long npts = track_starts[i+1] - track_start;

            NT max_val = std::numeric_limits<NT>::lowest();
            unsigned long max_id = 0;

            for (size_t j = 0; j < npts; ++j)
            {
                unsigned long id = track_start + j;
                NT val = psurface_wind[id];
                bool max_changed = val > max_val;
                max_val = max_changed ? val : max_val;
                max_id = max_changed ? id : max_id;
            }

            pcategory[i] = internal::classify_saphir_simpson(max_val);
            pmax_surface_wind[i] = max_val;
            pmax_surface_wind_id[i] = max_id;
        }
        )

    // location of the max surface wind
    p_teca_variant_array max_surface_wind_x = x->new_instance(n_tracks);
    p_teca_variant_array max_surface_wind_y = x->new_instance(n_tracks);
    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        start_x.get(),

        const NT *px = static_cast<const TT*>(x.get())->get();
        const NT *py = static_cast<const TT*>(y.get())->get();

        NT *pmax_surface_wind_x
            = static_cast<TT*>(max_surface_wind_x.get())->get();

        NT *pmax_surface_wind_y
            = static_cast<TT*>(max_surface_wind_y.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmax_surface_wind_id[i];
            pmax_surface_wind_x[i] = px[q];
            pmax_surface_wind_y[i] = py[q];
        }
        )

    // time of max surface wind
    p_teca_variant_array max_surface_wind_t = time->new_instance(n_tracks);
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        max_surface_wind_t.get(),
        const NT *ptime = static_cast<const TT*>(time.get())->get();

        NT *pmax_surface_wind_t
            = static_cast<TT*>(max_surface_wind_t.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmax_surface_wind_id[i];
            pmax_surface_wind_t[i] = ptime[q];
        }
        )

    // record the min sea level pressure
    p_teca_variant_array min_sea_level_pressure =
        sea_level_pressure->new_instance(n_tracks);

    p_teca_unsigned_long_array min_sea_level_pressure_id
        = teca_unsigned_long_array::New(n_tracks);

    unsigned long *pmin_sea_level_pressure_id
        = min_sea_level_pressure_id->get();

    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        min_sea_level_pressure.get(),
        const NT *psea_level_pressure =
            static_cast<const TT*>(sea_level_pressure.get())->get();

        NT *pmin_sea_level_pressure =
            static_cast<TT*>(min_sea_level_pressure.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long track_start = track_starts[i];
            unsigned long npts = track_starts[i+1] - track_start;

            NT min_val = std::numeric_limits<NT>::max();
            unsigned long min_id = 0;

            for (size_t j = 0; j < npts; ++j)
            {
                unsigned long q = track_start + j;
                NT val = psea_level_pressure[q];
                bool min_changed = val < min_val;
                min_val = min_changed ? val : min_val;
                min_id = min_changed ? q : min_id;
            }

            pmin_sea_level_pressure[i] = min_val;
            pmin_sea_level_pressure_id[i] = min_id;
        }
        )

    // location of the min sea level pressure
    p_teca_variant_array min_sea_level_pressure_x = x->new_instance(n_tracks);
    p_teca_variant_array min_sea_level_pressure_y = x->new_instance(n_tracks);
    TEMPLATE_DISPATCH_FP(teca_variant_array_impl,
        start_x.get(),

        const NT *px = static_cast<const TT*>(x.get())->get();
        const NT *py = static_cast<const TT*>(y.get())->get();

        NT *pmin_sea_level_pressure_x
            = static_cast<TT*>(min_sea_level_pressure_x.get())->get();

        NT *pmin_sea_level_pressure_y
            = static_cast<TT*>(min_sea_level_pressure_y.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmin_sea_level_pressure_id[i];
            pmin_sea_level_pressure_x[i] = px[q];
            pmin_sea_level_pressure_y[i] = py[q];
        }
        )

    // time of min sea level pressure
    p_teca_variant_array min_sea_level_pressure_t = time->new_instance(n_tracks);
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        min_sea_level_pressure_t.get(),
        const NT *ptime = static_cast<const TT*>(time.get())->get();

        NT *pmin_sea_level_pressure_t
            = static_cast<TT*>(min_sea_level_pressure_t.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = pmin_sea_level_pressure_id[i];
            pmin_sea_level_pressure_t[i] = ptime[q];
        }
        )

    // cyclogenisis, determine region of origin
    size_t n_regions = this->region_sizes.size();

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
            oss << "r" << i;
            rnames.push_back(oss.str());
        }
    }

    std::vector<std::string> rlnames(this->region_long_names);
    if (rnames.empty())
    {
        // generate names
        std::ostringstream oss;
        rnames.reserve(n_regions);
        for (size_t  i = 0; i < n_regions; ++i)
        {
            oss.str("");
            oss << "region_" << i;
            rlnames.push_back(oss.str());
        }
    }

    p_teca_int_array region_id = teca_int_array::New(n_tracks, -1);
    int *pregion_id = region_id->get();

    p_teca_string_array region_name = teca_string_array::New(n_tracks);
    std::string *pregion_name = region_name->get();

    p_teca_string_array region_long_name = teca_string_array::New(n_tracks);
    std::string *pregion_long_name = region_long_name->get();

    TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        x.get(),

        const NT *px = static_cast<const TT*>(x.get())->get();
        const NT *py = static_cast<const TT*>(y.get())->get();

        for (size_t i = 0; i < n_tracks; ++i)
        {
            unsigned long q = track_starts[i];

            double ptx = px[q];
            double pty = py[q];

            for (size_t j = 0; j < n_regions; ++j)
            {
                double *polyx = this->region_x_coordinates.data() + rstarts[j];
                double *polyy = this->region_y_coordinates.data() + rstarts[j];
                if (teca_geometry::point_in_poly(ptx, pty, polyx, polyy, this->region_sizes[j]))
                {
                    unsigned long rid = rids[j];
                    pregion_id[i] = rid;
                    pregion_name[i] = rnames[rid];
                    pregion_long_name[i] = rlnames[rid];
                    // early termination precludes a storm from being counted in
                    // multiple regions. if we want to allow overlapping regions
                    // this would need to change.
                    break;
                }
            }

            if (pregion_id[i] < 0)
            {
                TECA_WARNING("track " << i << " is not any of the regions!")
            }
        }
        )

    // construct the output
    p_teca_table out_table = teca_table::New();
    out_table->copy_metadata(in_table);

    out_table->append_column("track_id", out_ids);
    out_table->append_column("start_time", start_time);
    out_table->append_column("start_x", start_x);
    out_table->append_column("start_y", start_y);
    out_table->append_column("duration", duration);
    out_table->append_column("length", length);
    out_table->append_column("category", category);
    out_table->append_column("max_surface_wind", max_surface_wind);
    out_table->append_column("max_surface_wind_x", max_surface_wind_x);
    out_table->append_column("max_surface_wind_y", max_surface_wind_y);
    out_table->append_column("max_surface_wind_t", max_surface_wind_t);
    out_table->append_column("min_sea_level_pressure", min_sea_level_pressure);
    out_table->append_column("min_sea_level_pressure_x", min_sea_level_pressure_x);
    out_table->append_column("min_sea_level_pressure_y", min_sea_level_pressure_y);
    out_table->append_column("min_sea_level_pressure_t", min_sea_level_pressure_t);
    out_table->append_column("region_id", region_id);
    out_table->append_column("region_name", region_name);
    out_table->append_column("region_long_name", region_long_name);

    return out_table;
}
