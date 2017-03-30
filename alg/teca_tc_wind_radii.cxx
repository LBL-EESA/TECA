#include "teca_tc_wind_radii.h"

#include "teca_cartesian_mesh.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_coordinate_util.h"
#include "teca_table.h"
#include "teca_programmable_algorithm.h"
#include "teca_saffir_simpson.h"

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <sstream>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using std::cout;
using std::cerr;
using std::endl;

//#define TECA_DEBUG

// PIMPL idiom hides internals
class teca_tc_wind_radii::internals_t
{
public:
    internals_t();
    ~internals_t();

    void clear();

    teca_algorithm_output_port storm_pipeline_port; // pipeline that serves up tracks
    teca_metadata metadata;                         // cached metadata
    const_p_teca_table storm_table;                 // data structures that enable
    unsigned long number_of_storms;                 // random access into tracks
    std::vector<unsigned long> storm_counts;
    std::vector<unsigned long> storm_offsets;
    std::vector<unsigned long> storm_ids;

public:
    template <typename NT_MESH, typename NT_WIND>
    static int locate_critical_ids(
        NT_MESH *rad,               // bin centers
        NT_WIND *wind,              // wind speed at centers (max, avg, etc)
        unsigned int n_bins,        // length of profile arrays
        NT_MESH core_rad_max,       // max allowed distance to peak
        NT_WIND *crit_wind,         // speeds to calculate radius at
        unsigned int n_crit,        // number of critical values
        unsigned int *crit_ids,     // index of critical wind
        unsigned int &peak_id);     // index of peak wind

    // given two points (x1,y1), (x2,y2) defining a line
    // and a thrid point defining a horizontal line (*, yc)
    // compute the x value, (xc, yc)  where the lines
    // intersect.
    template <typename NT_MESH>
    static int compute_crossing(
        NT_MESH x1, NT_MESH y1, NT_MESH x2,
        NT_MESH y2, NT_MESH yc, NT_MESH &xc);

    // given the set of critical ids from locate critical ids
    // compute linear aproximation of the intersections with
    // critical wind speeds.
    template <typename NT_MESH, typename NT_WIND>
    static int compute_crossings(NT_MESH *rad, NT_WIND *wind,
        NT_WIND *crit_wind, unsigned int n_crit, unsigned int *crit_ids,
        NT_MESH *rcross);

    template <typename NT_MESH, typename NT_WIND>
    static NT_WIND compute_wind_speed(
            NT_MESH x, NT_MESH y, NT_WIND u, NT_WIND v);

    // initialize the vector with the default speeds to compute
    // radius at. These are the transitions of the Saffir-Simpson
    // scale.
    static void init_critical_wind_speeds(
        std::vector<double> &critical_wind_speeds);

    // binning operators, used to map Cartesian mesh onto
    // a radial mesh.
    template<typename NT> class bin_average;
    template<typename NT> class bin_max;

    // mapping functions. overloads are used to get around
    // the fact that templates with more than one argument
    // are difficult to use inside of the dispatch macro
    template<typename NT_MESH, typename NT_WIND,
        template<typename> class bin_operation_t>
    static p_teca_variant_array_impl<NT_WIND>
    compute_radial_profile(NT_MESH storm_x, NT_MESH storm_y,
        const NT_MESH *mesh_x, const NT_MESH *mesh_y,
        const NT_WIND *wind_u, const NT_WIND *wind_v,
        unsigned long nx, unsigned long ny, int number_of_bins,
        NT_MESH bin_width, NT_MESH max_radius,
        p_teca_variant_array_impl<NT_MESH> &rad_all,
        p_teca_variant_array_impl<NT_WIND> &wind_all);

    template<typename NT_MESH, typename NT_WIND>
    static p_teca_variant_array_impl<NT_WIND>
    compute_average_radial_profile(NT_MESH storm_x, NT_MESH storm_y,
        const NT_MESH *mesh_x, const NT_MESH *mesh_y,
        const NT_WIND *wind_u, const NT_WIND *wind_v,
        unsigned long nx, unsigned long ny, int number_of_bins,
        NT_MESH bin_width, NT_MESH max_radius,
        p_teca_variant_array_impl<NT_MESH> &rad_all,
        p_teca_variant_array_impl<NT_WIND> &wind_all)
    {
        return compute_radial_profile<NT_MESH, NT_WIND, bin_average>(
            storm_x, storm_y, mesh_x, mesh_y, wind_u, wind_v, nx, ny,
            number_of_bins, bin_width, max_radius, rad_all, wind_all);
    }

    template<typename NT_MESH, typename NT_WIND>
    static p_teca_variant_array_impl<NT_WIND>
    compute_max_radial_profile(NT_MESH storm_x, NT_MESH storm_y,
        const NT_MESH *mesh_x, const NT_MESH *mesh_y,
        const NT_WIND *wind_u, const NT_WIND *wind_v,
        unsigned long nx, unsigned long ny, int number_of_bins,
        NT_MESH bin_width, NT_MESH max_radius,
        p_teca_variant_array_impl<NT_MESH> &rad_all,
        p_teca_variant_array_impl<NT_WIND> &wind_all)
    {
        return compute_radial_profile<NT_MESH, NT_WIND, bin_max>(
            storm_x, storm_y, mesh_x, mesh_y, wind_u, wind_v, nx, ny,
            number_of_bins, bin_width, max_radius, rad_all, wind_all);
    }

    // function generate Python code to plot the radial profile
    // used for debuging only
    template<typename NT_MESH, typename NT_WIND>
    static void plot_radial_profile(std::ostream &ostr,
        unsigned long track_id, unsigned int k,
        p_teca_variant_array_impl<NT_MESH> rad_all,
        p_teca_variant_array_impl<NT_WIND> wind_all,
        p_teca_variant_array_impl<NT_MESH> rad,
        p_teca_variant_array_impl<NT_WIND> wind,
        const std::vector<NT_WIND> &crit_wind,
        const std::vector<unsigned int> &crit_ids,
        unsigned int peak_id,
        p_teca_variant_array_impl<NT_MESH> rcross);

};

template<typename NT>
class teca_tc_wind_radii::internals_t::bin_average
{
public:
    bin_average() = delete;
    bin_average(int nbins) : m_nbins(nbins)
    {
        m_vals = teca_variant_array_impl<NT>::New(nbins, NT());
        m_pvals = m_vals->get();

        m_count = teca_int_array::New(nbins, 0);
        m_pcount = m_count->get();
    }

    void operator()(int bin, NT val)
    {
        m_pvals[bin] += val;
        m_pcount[bin] += 1;
    }

    p_teca_variant_array_impl<NT> get_bin_values()
    {
        for (int i = 0; i < m_nbins; ++i)
            m_pvals[i] = m_pcount[i] ? m_pvals[i]/m_pcount[i] : m_pvals[i];
        return m_vals;
    }

private:
    p_teca_variant_array_impl<NT> m_vals;
    NT *m_pvals;
    p_teca_int_array m_count;
    int *m_pcount;
    int m_nbins;
};


template<typename NT>
class teca_tc_wind_radii::internals_t::bin_max
{
public:
    bin_max() = delete;
    bin_max(int nbins)
    {
        m_vals = teca_variant_array_impl<NT>::New(nbins, NT());
        m_pvals = m_vals->get();
    }

    void operator()(int bin, NT val)
    { m_pvals[bin] = std::max(m_pvals[bin], val); }

    p_teca_variant_array_impl<NT> get_bin_values()
    { return m_vals; }

private:
    p_teca_variant_array_impl<NT> m_vals;
    NT *m_pvals;
};


// --------------------------------------------------------------------------
teca_tc_wind_radii::internals_t::internals_t() : number_of_storms(0)
{}

// --------------------------------------------------------------------------
teca_tc_wind_radii::internals_t::~internals_t()
{}

// --------------------------------------------------------------------------
void teca_tc_wind_radii::internals_t::clear()
{
    this->metadata.clear();
    this->storm_table = nullptr;
    this->number_of_storms = 0;
    this->metadata.clear();
    this->storm_counts.clear();
    this->storm_offsets.clear();
    this->storm_ids.clear();
}

// --------------------------------------------------------------------------
template <typename NT_MESH, typename NT_WIND>
NT_WIND teca_tc_wind_radii::internals_t::compute_wind_speed(
    NT_MESH x, NT_MESH y, NT_WIND u, NT_WIND v)
{
#if defined(AZIMUTHAL_PROFILE)
    // azimuthal wind speed
    NT_MESH theta = std::atan2(y, x);
    NT_WIND w = std::fabs(-u*std::sin(theta) + v*std::cos(theta));
#else
    (void)x;
    (void)y;
    NT_WIND uu = u*u;
    NT_WIND vv = v*v;
    NT_WIND w = std::sqrt(uu + vv);
#endif
    return w;
}

// --------------------------------------------------------------------------
template <typename NT_MESH>
int teca_tc_wind_radii::internals_t::compute_crossing(NT_MESH x1,
    NT_MESH y1, NT_MESH x2, NT_MESH y2, NT_MESH yc, NT_MESH &xc)
{
    NT_MESH D = x1 - x2;
#if defined(TECA_DEBUG)
    if (std::fabs(D) <= NT_MESH(1.0e-6))
    {
        TECA_ERROR("cooincident points. cannot compute slope intercept")
        xc = x2;
        return -1;
    }
#endif
    NT_MESH m = (y1 - y2)/D;
    NT_MESH b = (x1*y2 - x2*y1)/D;
    xc = (yc - b)/m;
    return 0;
}

// --------------------------------------------------------------------------
template <typename NT_MESH, typename NT_WIND>
int teca_tc_wind_radii::internals_t::compute_crossings(NT_MESH *rad,
    NT_WIND *wind, NT_WIND *crit_wind, unsigned int n_crit,
    unsigned int *crit_ids, NT_MESH *rcross)
{
    // zero out outputs
    memset(rcross, 0, n_crit*sizeof(NT_MESH));

    // for each critical speed where a radial crossing was detected
    // solve for the interecpt of the linear approximtion of the
    // radial profile and the horizontal line definied by the critical
    // wind value
    for (unsigned int i = 0; i < n_crit; ++i)
    {
        if (crit_ids[i])
        {
            // by construction we know crossing is in between these ids
            // the ids name 2 points defining a line guaranteed to intercect
            // the horizontal line defined by the critical wind speed
            unsigned int q2 = crit_ids[i];
            unsigned int q1 = q2-1;

            compute_crossing<NT_MESH>(
                rad[q1], wind[q1], rad[q2], wind[q2],
                crit_wind[i], rcross[i]);
        }
    }

    return 0;
}

// --------------------------------------------------------------------------
template <typename NT_MESH, typename NT_WIND>
int teca_tc_wind_radii::internals_t::locate_critical_ids(
    NT_MESH *rad, NT_WIND *wind, unsigned int n_bins, NT_MESH core_rad_max,
    NT_WIND *crit_wind, unsigned int n_crit, unsigned int *crit_ids,
    unsigned int &peak_id)
{
    // first zero out everything
    for (unsigned int i = 0; i < n_crit; ++i)
        crit_ids[i] = 0;

    // locate the peak wind and peak rad
    peak_id = 0;
    for (unsigned int i = 1; i < n_bins; ++i)
        peak_id = wind[i] > wind[peak_id] ? i : peak_id;

    // peak wind speed should be close to the storm center
    // inheritted from the GFDL algorithm requirements
    if (rad[peak_id] > core_rad_max)
    {
        TECA_WARNING("Peak wind speed is outside of the core "
            << rad[peak_id] << " > " << core_rad_max)
        peak_id = std::numeric_limits<unsigned int>::max();
        return -1;
    }

    // locate the critical values
    for (unsigned int i = 0; i < n_crit; ++i)
    {
        // skip when search is impossible
        if (crit_wind[i] >= wind[peak_id])
            continue;

        // find the first less or equal to the critical value
        // from the peak
        for (unsigned int j = peak_id; (j < n_bins) && !crit_ids[i]; ++j)
            crit_ids[i] = wind[j] < crit_wind[i] ? j : 0;
    }

    return 0;
}

// --------------------------------------------------------------------------
template<typename NT_MESH, typename NT_WIND,
    template<typename> class bin_operation_t>
p_teca_variant_array_impl<NT_WIND>
teca_tc_wind_radii::internals_t::compute_radial_profile(NT_MESH storm_x,
    NT_MESH storm_y, const NT_MESH *mesh_x, const NT_MESH *mesh_y,
    const NT_WIND *wind_u, const NT_WIND *wind_v, unsigned long nx,
    unsigned long ny, int number_of_bins, NT_MESH bin_width,
    NT_MESH max_radius, p_teca_variant_array_impl<NT_MESH> &rad_all,
    p_teca_variant_array_impl<NT_WIND> &wind_all)
{
#if defined(TECA_DEBUG)
    unsigned long nxy = nx*ny;

    rad_all = teca_variant_array_impl<NT_MESH>::New();
    rad_all->reserve(nxy);

    wind_all = teca_variant_array_impl<NT_WIND>::New();
    wind_all->reserve(nxy);
#else
    (void)rad_all;
    (void)wind_all;
#endif

    // construct an instance of the binning operator
    bin_operation_t<NT_WIND> bin_op(number_of_bins);

    // for each grid point compute radial distance to storm center
    for (unsigned long j = 0; j < ny; ++j)
    {
        unsigned long q = j*nx;
        NT_MESH y = mesh_y[j] - storm_y;
        NT_MESH yy = y*y;
        for (unsigned long i = 0; i < nx; ++i)
        {
            // radius
            NT_MESH x = mesh_x[i] - storm_x;
            NT_MESH xx = x*x;
            NT_MESH r = std::sqrt(xx + yy);

            if (r <= max_radius)
            {
                // compute wind speed at the grid point
                NT_WIND w = teca_tc_wind_radii::internals_t::
                    compute_wind_speed(x, y, wind_u[q+i], wind_v[q+i]);

                // sample it onto the discrete radial mesh
                unsigned int bin = static_cast<unsigned int>(r/bin_width);
                bin_op(bin, w);

#if defined(TECA_DEBUG)
                rad_all->append(r);
                wind_all->append(w);
#endif
            }
        }
    }

    return bin_op.get_bin_values();
}

// --------------------------------------------------------------------------
void teca_tc_wind_radii::internals_t::init_critical_wind_speeds(
    std::vector<double> &critical_wind_speeds)
{
    // critical wind speeds are the thresholds of Saffir-Simpson
    // scale, starting at tropical depression -1 up to cat 5 storm
    double wind_crit_mps[6] = {
        teca_saffir_simpson::get_upper_bound_mps<double>(-1),
        teca_saffir_simpson::get_upper_bound_mps<double>(0),
        teca_saffir_simpson::get_upper_bound_mps<double>(1),
        teca_saffir_simpson::get_upper_bound_mps<double>(2),
        teca_saffir_simpson::get_upper_bound_mps<double>(3),
        teca_saffir_simpson::get_upper_bound_mps<double>(4)};

    critical_wind_speeds.assign(wind_crit_mps, wind_crit_mps + 6);
}

// --------------------------------------------------------------------------
template<typename NT_MESH, typename NT_WIND>
void teca_tc_wind_radii::internals_t::plot_radial_profile(
    std::ostream &ostr, unsigned long storm_id, unsigned int k,
    p_teca_variant_array_impl<NT_MESH> rad_all,
    p_teca_variant_array_impl<NT_WIND> wind_all,
    p_teca_variant_array_impl<NT_MESH> rad,
    p_teca_variant_array_impl<NT_WIND> wind,
    const std::vector<NT_WIND> &crit_wind,
    const std::vector<unsigned int> &crit_ids,
    unsigned int peak_id,
    p_teca_variant_array_impl<NT_MESH> rcross)
{
    // generate Python code that can plot the radial profile
    ostr << "rad_all = [";
    rad_all->to_stream(ostr);
    ostr << "]" << endl;

    ostr << "wind_all = [";
    wind_all->to_stream(ostr);
    ostr << "]" << endl;

    ostr << "rad = [";
    rad->to_stream(ostr);
    ostr << "]" << endl;

    ostr << "wind = [";
    wind->to_stream(ostr);
    ostr << "]" << endl;

    ostr << "rcross = [";
    rcross->to_stream(ostr);
    ostr << "]" << endl;

    unsigned int n_crit_vals = crit_wind.size();
    ostr << "crit_wind_req = [" << crit_wind[0];
    for (unsigned int i = 1; i < n_crit_vals; ++i)
        ostr << ", " << crit_wind[i];
    ostr << "]" << endl;

    ostr << "crit_rad = [" << rad->get(crit_ids[0]);
    for (unsigned int i = 1; i < n_crit_vals; ++i)
        ostr << ", " << rad->get(crit_ids[i]);
    ostr << "]" << endl;

    ostr << "crit_wind_got = [" << wind->get(crit_ids[0]);
    for (unsigned int i = 1; i < n_crit_vals; ++i)
        ostr << ", " << wind->get(crit_ids[i]);
    ostr << "]" << endl;

    ostr << "peak_rad = " << rad->get(peak_id) << endl
        << "peak_wind = " << wind->get(peak_id) << endl;

    ostr << "dom = [0, max(rad_all)]" << endl
        << "rng = [0, 1.1*max(" << crit_wind.back()
        << ", " << wind->get(peak_id) << ")]" << endl;

    ostr << "fig = mpl.figure()" << endl;

    // crit vals
    for (unsigned int i = 0; i < n_crit_vals; ++i)
        ostr << "mpl.plot(dom, [crit_wind_req[" << i << "]"
            << ", crit_wind_req[" << i << "]], 'r--', alpha=0.5)"
            << endl;

    // scatter plot of inputs
    ostr << "mpl.plot(rad_all, wind_all, '.', markerfacecolor='none',"
        << " markeredgecolor='#000000', alpha=0.15)" << endl;

    // line plot wind profile
    ostr << "mpl.plot(rad, wind, 'k-', linewidth=2)" << endl
        << "mpl.plot(rad, wind, 'k.')" << endl;

    // critical radii
    for (unsigned int i = 0; i < n_crit_vals; ++i)
        ostr << "mpl.plot([rcross[" << i << "]]*2, [0, crit_wind_req[" << i << "]],"
            << " 'b--', alpha=0.5)" << endl;

    ostr << "mpl.plot(crit_rad, crit_wind_got, 'b+',"
        << " markerfacecolor='none', markeredgewidth=2)" << endl;

    ostr << "mpl.plot(rcross, crit_wind_req, 'bo',"
        << " markerfacecolor='y', markeredgewidth=2)" << endl;

    // peak raddii
    ostr << "mpl.plot([peak_rad]*2, [0, peak_wind],"
        << " 'b--', alpha=0.5)" << endl;

    ostr << "mpl.plot(peak_rad, peak_wind, 'b^',"
        << " markerfacecolor='none', markeredgewidth=2)" << endl;

    // format the plot
    ostr << "ax = mpl.gca()" << endl
        << "yl = ax.get_ylim()" << endl
        << "xl = ax.get_xlim()" << endl
        << "yl = [0, yl[1]]" << endl
        << "xl = [0, int(xl[1])]" << endl
        << "mpl.title('radial profile track=" << storm_id << " step=" << k << "')" << endl
        << "mpl.xlabel('dist to storm center in deg lat')" << endl
        << "mpl.ylabel('wind speed in m/s')" << endl
        << "mpl.grid(True)" << endl
        << "mpl.xlim(dom)" << endl
        << "mpl.ylim(rng)" << endl;

    // save it
    ostr << "mpl.savefig('radial_wind_profile_"
        << std::setfill('0') << std::setw(5) << storm_id << "_"
        << std::setfill('0') << std::setw(5) << k << ".png')"
        << endl;

    ostr << "mpl.close(fig)" << endl
        << "sys.stderr.write('*')" << endl;
}


// --------------------------------------------------------------------------
teca_tc_wind_radii::teca_tc_wind_radii() : storm_id_column("track_id"),
    storm_x_coordinate_column("lon"), storm_y_coordinate_column("lat"),
    storm_wind_speed_column("surface_wind"), storm_time_column("time"),
    wind_u_variable("UBOT"), wind_v_variable("VBOT"),
    critical_wind_speeds({
        teca_saffir_simpson::get_upper_bound_mps<double>(-1),
        teca_saffir_simpson::get_upper_bound_mps<double>(0),
        teca_saffir_simpson::get_upper_bound_mps<double>(1),
        teca_saffir_simpson::get_upper_bound_mps<double>(2),
        teca_saffir_simpson::get_upper_bound_mps<double>(3),
        teca_saffir_simpson::get_upper_bound_mps<double>(4)}),
    search_radius(6.0), core_radius(std::numeric_limits<double>::max()),
    number_of_radial_bins(32), profile_type(PROFILE_AVERAGE)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);

    this->internals = new teca_tc_wind_radii::internals_t;
}

// --------------------------------------------------------------------------
teca_tc_wind_radii::~teca_tc_wind_radii()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_tc_wind_radii::get_properties_description(const std::string &prefix,
    options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_tc_wind_radii":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, storm_id_column,
            "name of the column containing unique ids of the storms")
        TECA_POPTS_GET(std::string, prefix, storm_x_coordinate_column,
            "name of the column to create storm x coordinates from")
        TECA_POPTS_GET(std::string, prefix, storm_y_coordinate_column,
            "name of the column to create storm y coordinates from")
        TECA_POPTS_GET(std::string, prefix, storm_time_column,
            "name of the column to create storm times from")
        TECA_POPTS_GET(std::string, prefix, wind_u_variable,
            "name of the variable containing u component of wind")
        TECA_POPTS_GET(std::string, prefix, wind_v_variable,
            "name of the variable containing v component of wind")
        TECA_POPTS_GET(double, prefix, search_radius,
            "defines the radius of the search space in deg lat")
        TECA_POPTS_GET(double, prefix, core_radius,
            "defines the radius inside which the core is expected in deg lat")
        TECA_POPTS_MULTI_GET(std::vector<double>, prefix, critical_wind_speeds,
            "sets the wind speeds to compute radii at")
        TECA_POPTS_GET(int, prefix, number_of_radial_bins,
            "sets the number of bins to discretize in the radial direction")
        TECA_POPTS_GET(int, prefix, profile_type,
            "determines how profile values are computed. for PROFILE_MAX=0 "
            "the max wind speed over each interval is used, for PROFILE_AVERAGE=1 "
            "the average wind speed over the interval is used.")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_tc_wind_radii::set_properties(const std::string &prefix,
    variables_map &opts)
{
    TECA_POPTS_SET(opts, std::string, prefix, storm_id_column)
    TECA_POPTS_SET(opts, std::string, prefix, storm_x_coordinate_column)
    TECA_POPTS_SET(opts, std::string, prefix, storm_y_coordinate_column)
    TECA_POPTS_SET(opts, std::string, prefix, storm_time_column)
    TECA_POPTS_SET(opts, std::string, prefix, wind_u_variable)
    TECA_POPTS_SET(opts, std::string, prefix, wind_v_variable)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, critical_wind_speeds)
    TECA_POPTS_SET(opts, double, prefix, search_radius)
    TECA_POPTS_SET(opts, double, prefix, core_radius)
    TECA_POPTS_SET(opts, int, prefix, number_of_radial_bins)
    TECA_POPTS_SET(opts, int, prefix, profile_type)
}
#endif

// --------------------------------------------------------------------------
void teca_tc_wind_radii::set_input_connection(unsigned int id,
        const teca_algorithm_output_port &port)
{
    if (id == 0)
        this->internals->storm_pipeline_port = port;
    else
        this->teca_algorithm::set_input_connection(0, port);
}

// --------------------------------------------------------------------------
void teca_tc_wind_radii::set_modified()
{
    // clear cached metadata before forwarding on to
    // the base class.
    this->internals->clear();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
teca_metadata teca_tc_wind_radii::teca_tc_wind_radii::get_output_metadata(
    unsigned int port, const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_tc_wind_radii::get_output_metadata" << endl;
#endif
    (void)port;
    (void)input_md;

    if (this->internals->storm_table)
        return this->internals->metadata;

    // execute the pipeline that retruns table of tracks
    const_p_teca_dataset storm_data;

    p_teca_programmable_algorithm capture_storm_data
        = teca_programmable_algorithm::New();

    capture_storm_data->set_input_connection(this->internals->storm_pipeline_port);

    capture_storm_data->set_execute_callback(
        [&storm_data] (unsigned int, const std::vector<const_p_teca_dataset> &in_data,
     const teca_metadata &) -> const_p_teca_dataset
     {
         storm_data = in_data[0];
         return nullptr;
     });

    capture_storm_data->update();

    int rank = 0;
#if defined(TECA_HAS_MPI)
    int is_init = 0;
    MPI_Initialized(&is_init);
    if (is_init)
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    // validate the table
    if (rank == 0)
    {
        // did the pipeline run successfully
        const_p_teca_table storm_table =
            std::dynamic_pointer_cast<const teca_table>(storm_data);

        if (!storm_table)
        {
            TECA_ERROR("metadata pipeline failure")
        }

        // column need to build random access data structures
        const_p_teca_variant_array storm_ids =
            storm_table->get_column(this->storm_id_column);

        if (!storm_ids)
        {
            TECA_ERROR("storm index column \""
            << this->storm_id_column << "\" not found")
        }
        // these columns are needed to compute the storm size
        else
        if (!storm_table->has_column(this->storm_x_coordinate_column))
        {
            TECA_ERROR("storm x coordinates column \""
                << this->storm_x_coordinate_column << "\" not found")
        }
        else
        if (!storm_table->has_column(this->storm_y_coordinate_column))
        {
            TECA_ERROR("storm y coordinates column \""
                << this->storm_y_coordinate_column << "\" not found")
        }
        else
        if (!storm_table->has_column(this->storm_wind_speed_column))
        {
            TECA_ERROR("storm wind speed column \""
                << this->storm_wind_speed_column << "\" not found")
        }
        else
        if (!storm_table->has_column(this->storm_time_column))
        {
            TECA_ERROR("storm time column \""
                << this->storm_time_column << "\" not found")
        }
        // things are ok, take a reference
        else
        {
            this->internals->storm_table = storm_table;
        }
    }

    // distribute the table to all processes
#if defined(TECA_HAS_MPI)
    if (is_init)
    {
        teca_binary_stream bs;
        if (this->internals->storm_table && (rank == 0))
            this->internals->storm_table->to_stream(bs);
        bs.broadcast();
        if (bs && (rank != 0))
        {
           p_teca_table tmp = teca_table::New();
           tmp->from_stream(bs);
           this->internals->storm_table = tmp;
        }
    }
#endif

    // build random access data structures
    const_p_teca_variant_array storm_ids =
        this->internals->storm_table->get_column(this->storm_id_column);

    TEMPLATE_DISPATCH_I(const teca_variant_array_impl,
        storm_ids.get(),

        const NT *pstorm_ids = dynamic_cast<TT*>(storm_ids.get())->get();

        teca_coordinate_util::get_table_offsets(pstorm_ids,
            this->internals->storm_table->get_number_of_rows(),
            this->internals->number_of_storms, this->internals->storm_counts,
            this->internals->storm_offsets, this->internals->storm_ids);
        )

    // must have at least one time storm
    if (this->internals->number_of_storms < 1)
    {
        TECA_ERROR("Invalid index \"" << this->storm_id_column << "\"")
        this->internals->clear();
        return teca_metadata();
    }

    // report about the number of steps, this is all that
    // is needed to run in parallel over time steps.
    this->internals->metadata.clear();
    this->internals->metadata.insert(
        "number_of_time_steps", this->internals->number_of_storms);

    return this->internals->metadata;
}

// --------------------------------------------------------------------------
std::vector<teca_metadata> teca_tc_wind_radii::get_upstream_request(
    unsigned int port, const std::vector<teca_metadata> &input_md,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << " teca_tc_wind_radii::get_upstream_request" << endl;
#endif
    (void)port;
    (void)input_md;

    // get the mesh bounds
    teca_metadata mesh_coords;
    input_md[0].get("coordinates", mesh_coords);

    const_p_teca_variant_array mesh_x = mesh_coords.get("x");

    unsigned long mesh_ext[6];
    input_md[0].get("whole_extent", mesh_ext, 6);

    double mesh_x0, mesh_x1;
    mesh_x->get(mesh_ext[0], mesh_x0);
    mesh_x->get(mesh_ext[1], mesh_x1);

    // get id of storm id being requested
    unsigned long map_id = 0;
    request.get("time_step", map_id);

    // get the storm track data, location and time
    unsigned long id_ofs = this->internals->storm_offsets[map_id];
    unsigned long n_ids = this->internals->storm_counts[map_id];

    const_p_teca_variant_array
    x_coordinates = this->internals->storm_table->get_column
            (this->storm_x_coordinate_column);

    const_p_teca_variant_array
    y_coordinates = this->internals->storm_table->get_column
            (this->storm_y_coordinate_column);

    const_p_teca_variant_array
    times = this->internals->storm_table->get_column
            (this->storm_time_column);

    // construct the base request
     std::vector<std::string> arrays
         ({this->wind_u_variable, this->wind_v_variable});

    teca_metadata base_req;
    base_req.insert("arrays", arrays);

    std::vector<teca_metadata> up_reqs(n_ids, base_req);

    // request the tile of dimension search radius centered on the
    // storm at this instant
    unsigned long n_incomplete = 0;
    TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        x_coordinates.get(),
        // for each point in track compute the bounding box needed
        // for the wind profile
        const NT *px = static_cast<TT*>(x_coordinates.get())->get();
        const NT *py = static_cast<TT*>(y_coordinates.get())->get();
        for (unsigned long i = 0; i < n_ids; ++i)
        {
            // TODO account for poleward longitude convergence
            NT x = px[i+id_ofs];
            NT y = py[i+id_ofs];
            NT r = static_cast<NT>(this->search_radius);

            NT x0 = x - r;
            NT x1 = x + r;
            NT y0 = y - r;
            NT y1 = y + r;

            // TODO -- implment periodic bc
            if ((x0 < NT(mesh_x0)) || (x1 > NT(mesh_x1)))
            {
                TECA_WARNING("In track " << map_id << " point " << i <<
                    " requires data across periodic boundary on ["
                    << x0 << ", " << x1 << ", " << y0 << ", " << y1 << "]")

                // clamp to the valid bounds
                x0 = std::max(NT(mesh_x0), x0);
                x1 = std::min(NT(mesh_x1), x1);

                ++n_incomplete;
            }

            // request the needed subset
            std::vector<double> bounds({x0, x1, y0, y1, 0.0, 0.0});
            up_reqs[i].insert("bounds", bounds);
        }
        )

    // give a summary of incomplete profiles
    if (n_incomplete)
    {
        TECA_WARNING("Profiles for " << n_incomplete << " of " << n_ids
            << " (" << ((double)n_incomplete)/n_ids << ") in track " << map_id
            << " are incomplete")
    }

    // request the specific time needed
    TEMPLATE_DISPATCH(const teca_variant_array_impl,
        times.get(),
        const NT *pt = static_cast<TT*>(times.get())->get();
        for (unsigned long i = 0; i < n_ids; ++i)
            up_reqs[i].insert("time", pt[i+id_ofs]);
        )

#ifdef TECA_DEBUG
   for (unsigned long i = 0; i < n_ids; ++i)
   {
       cerr << "req " << i << " = ";
       up_reqs[i].to_stream(cerr);
       cerr << endl;
   }
#endif

    return up_reqs;
}

// --------------------------------------------------------------------------
const_p_teca_dataset teca_tc_wind_radii::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << " teca_tc_wind_radii::execute" << endl;
    cout << "import matplotlib.pyplot as mpl" << endl
        << "import sys" << endl;
#endif
    (void)port;

    // get id of storm id being requested
    unsigned long storm_id = 0;
    request.get("time_step", storm_id);

    // for random access into the specific track
    unsigned long ofs = this->internals->storm_offsets[storm_id];
    unsigned long npts = this->internals->storm_counts[storm_id];

    // get strom track positions
    const_p_teca_variant_array storm_x =
        this->internals->storm_table->get_column(this->storm_x_coordinate_column);

    const_p_teca_variant_array storm_y =
        this->internals->storm_table->get_column(this->storm_y_coordinate_column);

    // allocate output columns
    unsigned int n_crit_vals = this->critical_wind_speeds.size();
    std::vector<p_teca_double_array> crit_radii(n_crit_vals);

    for (unsigned int i = 0; i < n_crit_vals; ++i)
        crit_radii[i] = teca_double_array::New(npts, 0.0);

    p_teca_double_array peak_radius = teca_double_array::New(npts, 0.0);
    p_teca_double_array peak_wind = teca_double_array::New(npts, 0.0);

    // compute radius at each point in time along the storm track
    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        storm_x.get(), _STORM,

        // get the storm centers
        const NT_STORM *pstorm_x = static_cast<TT_STORM*>(storm_x.get())->get();
        const NT_STORM *pstorm_y = static_cast<TT_STORM*>(storm_y.get())->get();

        // for each time instance in the storm compute the storm radius
        for (unsigned long k = 0; k < npts; ++k)
        {
            // get the kth mesh
            const_p_teca_cartesian_mesh mesh
                = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[k]);

            if (!mesh)
            {
                TECA_ERROR("input " << k << " is empty or not a cartesian mesh")
                return nullptr;
            }

            // and mesh coords.
            const_p_teca_variant_array mesh_x = mesh->get_x_coordinates();
            const_p_teca_variant_array mesh_y = mesh->get_y_coordinates();

            double t = 0.0;
            mesh->get_time(t);

            NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
                mesh_x.get(), _MESH,

                const NT_MESH *pmesh_x = static_cast<TT_MESH*>(mesh_x.get())->get();
                const NT_MESH *pmesh_y = static_cast<TT_MESH*>(mesh_y.get())->get();

                unsigned long nx = mesh_x->size();
                unsigned long ny = mesh_y->size();

                // construct radial discretization
                p_teca_variant_array_impl<NT_MESH> radius =
                    teca_variant_array_impl<NT_MESH>::New(this->number_of_radial_bins);

                NT_MESH max_radius = static_cast<NT_MESH>(this->search_radius);

                NT_MESH dr = max_radius/static_cast<NT_MESH>(this->number_of_radial_bins);
                NT_MESH dr_half = dr/NT_MESH(2);

                NT_MESH *pr = radius->get();

                for (int i = 0; i < this->number_of_radial_bins; ++i)
                    pr[i] = dr_half + static_cast<NT_MESH>(i)*dr;

                // get the wind components on the input mesh
                const_p_teca_variant_array wind_u =
                    mesh->get_point_arrays()->get(this->wind_u_variable);

                const_p_teca_variant_array wind_v =
                    mesh->get_point_arrays()->get(this->wind_v_variable);

                NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
                    wind_u.get(), _WIND,

                    const NT_WIND *pwu = static_cast<TT_WIND*>(wind_u.get())->get();
                    const NT_WIND *pwv = static_cast<TT_WIND*>(wind_v.get())->get();

                    // get the kth storm center
                    NT_MESH sx = static_cast<NT_MESH>(pstorm_x[k+ofs]);
                    NT_MESH sy = static_cast<NT_MESH>(pstorm_y[k+ofs]);

                    // compute the radial profile
                    p_teca_variant_array_impl<NT_MESH> rad_all;
                    p_teca_variant_array_impl<NT_WIND> wind_all;
                    p_teca_variant_array_impl<NT_WIND> wind;

                    switch (this->profile_type)
                    {
                    case PROFILE_AVERAGE:
                        wind = teca_tc_wind_radii::internals_t::compute_average_radial_profile
                                (sx,sy, pmesh_x, pmesh_y, pwu, pwv, nx, ny,
                                this->number_of_radial_bins, dr, max_radius,
                                rad_all, wind_all);
                        break;
                    case PROFILE_MAX:
                        wind = teca_tc_wind_radii::internals_t::compute_max_radial_profile
                                (sx,sy, pmesh_x, pmesh_y, pwu, pwv, nx, ny,
                                this->number_of_radial_bins, dr, max_radius,
                                rad_all, wind_all);
                        break;
                    default:
                        TECA_ERROR("Invalid profile type \"" << this->profile_type << "\"")
                        return nullptr;
                    }

                    // allocate temp for results
                    unsigned int peak_id = 0;
                    std::vector<unsigned int> crit_ids(n_crit_vals, 0u);
                    std::vector<NT_WIND> crit_wind(
                        this->critical_wind_speeds.begin(),
                        this->critical_wind_speeds.end());

                    // compute the offsets of the critical radii
                    NT_WIND *pw = wind->get();
                    teca_tc_wind_radii::internals_t::locate_critical_ids(
                        pr, pw, this->number_of_radial_bins,
                        static_cast<NT_MESH>(this->core_radius),
                        crit_wind.data(), n_crit_vals, crit_ids.data(),
                        peak_id);

                    // compute the intercepts with the critical wind speeds
                    p_teca_variant_array_impl<NT_MESH> rcross =
                        teca_variant_array_impl<NT_MESH>::New(n_crit_vals, NT_MESH());
                    NT_MESH *prcross = rcross->get();
                    teca_tc_wind_radii::internals_t::compute_crossings(pr, pw,
                        crit_wind.data(), n_crit_vals, crit_ids.data(), prcross);

                    // record critical radii
                    for (unsigned int i = 0; i < n_crit_vals; ++i)
                            crit_radii[i]->set(k, prcross[i]);

                    // record peak radius and peak wind speed
                    peak_radius->set(k,
                        peak_id == std::numeric_limits<unsigned int>::max() ?
                        0 : pr[peak_id]);

                    peak_wind->set(k,
                        peak_id == std::numeric_limits<unsigned int>::max() ?
                        0 : pw[peak_id]);

#if defined(TECA_DEBUG)
                    teca_tc_wind_radii::internals_t::plot_radial_profile(
                        std::cout, storm_id, k, rad_all, wind_all, radius,
                        wind, crit_wind, crit_ids, peak_id, rcross);
#endif
                    )
                )
        }
        )

    // pass the strom track through
    p_teca_table output = teca_table::New();
    output->copy(this->internals->storm_table, ofs, npts+ofs-1);

    // add the critial radii
    for (unsigned int i = 0; i < n_crit_vals; ++i)
    {
        std::ostringstream oss;
        oss << "wind_radius_" << i;
        output->append_column(oss.str(), crit_radii[i]);
    }

    // add the peak radii and wind speed
    output->append_column("peak_radius", peak_radius);
    output->append_column("peak_wind_speed", peak_wind);

    // add the critical wind speed values to the metadata
    output->get_metadata().insert(
        "critical_wind_speeds", this->critical_wind_speeds);

    return output;
}
