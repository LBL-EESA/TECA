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

// #define TECA_DEBUG

class teca_tc_wind_radii::internals_t
{
public:
    internals_t();
    ~internals_t();

    void clear();

    teca_algorithm_output_port track_pipeline_port; // pipeline that serves up tracks
    teca_metadata metadata;                         // cached metadata
    const_p_teca_table track_table;                 // data structures that enable
    unsigned long number_of_storms;                 // random access into tracks
    std::vector<unsigned long> track_counts;
    std::vector<unsigned long> track_offsets;
    std::vector<unsigned long> track_ids;

public:
    // copies the array into a new one with each element duplicated
    // n_per times.
    static p_teca_variant_array copy_expand_array(
        const_p_teca_variant_array va_in, unsigned long n_per);

    // estimate radial sampling paramteres based on the input mesh
    template<typename NT_MESH>
    static int compute_sampling_parameters(const NT_MESH *pmx,
        const NT_MESH *pmy, unsigned long nx, unsigned long ny,
        NT_MESH search_radius, unsigned int &r_resolution,
        NT_MESH &core_radius);

    // find the indices where starting from the peak the first
    // value of wind is less than the critical value.
    template <typename NT_MESH, typename NT_WIND>
    static int locate_critical_ids(NT_MESH *r, NT_WIND *w, NT_WIND *w_crit,
        unsigned int nr, unsigned int nt, unsigned int n_crit,
        unsigned int *crit_ids, unsigned int *peak_id);

    // given two points (x1,y1), (x2,y2) defining a line
    // and a thrid point defining a horizontal line (*, yc)
    // compute the x value, (xc, yc)  where the lines
    // intersect.
    template <typename NT_MESH>
    static int compute_crossing(NT_MESH x1, NT_MESH y1,
        NT_MESH x2, NT_MESH y2, NT_MESH yc, NT_MESH &xc);

    // given the set of critical ids from locate critical ids
    // compute linear aproximation of the intersections with
    // critical wind speeds.
    template <typename NT_MESH, typename NT_WIND>
    static int compute_crossings(NT_MESH *r, NT_WIND *w, NT_WIND *w_crit,
    unsigned int nr, unsigned int nt, unsigned int n_crit,
    unsigned int *crit_ids, NT_MESH *r_cross);

    template <typename NT_MESH, typename NT_WIND>
    static NT_WIND compute_wind_speed(NT_MESH x,
        NT_MESH y, NT_WIND u, NT_WIND v);

    // initialize the vector with the default speeds to compute
    // radius at. These are the transitions of the Saffir-Simpson
    // scale.
    static void init_critical_wind_speeds
        (std::vector<double> &critical_wind_speeds);

    // compute the wind radii at the given critical wind speeds from
    // nt radial profiles each with resolution of nr comosed of wind
    // wu,wv components defined on mesh mx,my centered at sx,sy.
    template <typename NT_MESH, typename NT_WIND>
    static int compute_wind_radii(unsigned int k, unsigned long track_id,
        int profile_type, unsigned int nr, unsigned int nt, NT_MESH r_core,
        NT_MESH r_max, NT_MESH sx, NT_MESH sy, const NT_MESH *mx,
        const NT_MESH *my, const NT_WIND *wu, const NT_WIND *wv,
        unsigned long nx, unsigned long ny, NT_WIND *w_crit,
        unsigned int n_crit, p_teca_variant_array_impl<NT_MESH> &r_crit,
        p_teca_variant_array_impl<NT_MESH> &r_peak,
        p_teca_variant_array_impl<NT_WIND> &w_peak);

    // binning operators, used to map Cartesian lon,lat mesh onto
    // a r,theta mesh.
    template<typename NT> class bin_average;
    template<typename NT> class bin_max;

    // mapping functions. overloads are used to get around
    // the fact that templates with more than one argument
    // are difficult to use inside of the dispatch macro
    template<typename NT_MESH, typename NT_WIND,
        template<typename> class bin_operation_t>
    static p_teca_variant_array_impl<NT_WIND>
    compute_radial_profile(NT_MESH sx, NT_MESH sy, const NT_MESH *mx,
        const NT_MESH *my, const NT_WIND *wu, const NT_WIND *wv,
        unsigned long nx, unsigned long ny, unsigned int nr, unsigned int nt,
        NT_MESH dr, NT_MESH dt, NT_MESH r_core, NT_MESH r_max,
        p_teca_variant_array_impl<NT_MESH> &r_all,
        p_teca_variant_array_impl<NT_MESH> &theta_all,
        p_teca_variant_array_impl<NT_WIND> &w_all);

    template<typename NT_MESH, typename NT_WIND>
    static p_teca_variant_array_impl<NT_WIND>
    compute_average_radial_profile(NT_MESH sx, NT_MESH sy, const NT_MESH *mx,
        const NT_MESH *my, const NT_WIND *wu, const NT_WIND *wv,
        unsigned long nx, unsigned long ny, unsigned int nr, unsigned int nt,
        NT_MESH dr, NT_MESH dt, NT_MESH r_core, NT_MESH r_max,
        p_teca_variant_array_impl<NT_MESH> &r_all,
        p_teca_variant_array_impl<NT_MESH> &theta_all,
        p_teca_variant_array_impl<NT_WIND> &w_all)
    {
        return compute_radial_profile<NT_MESH, NT_WIND, bin_average>(
            sx, sy, mx, my, wu, wv, nx, ny, nr, nt, dr, dt, r_core, r_max,
            r_all, theta_all, w_all);
    }

    template<typename NT_MESH, typename NT_WIND>
    static p_teca_variant_array_impl<NT_WIND>
    compute_max_radial_profile(NT_MESH sx, NT_MESH sy, const NT_MESH *mx,
        const NT_MESH *my, const NT_WIND *wu, const NT_WIND *wv,
        unsigned long nx, unsigned long ny, unsigned int nr, unsigned int nt,
        NT_MESH dr, NT_MESH dt, NT_MESH r_core, NT_MESH r_max,
        p_teca_variant_array_impl<NT_MESH> &r_all,
        p_teca_variant_array_impl<NT_MESH> &theta_all,
        p_teca_variant_array_impl<NT_WIND> &w_all)
    {
        return compute_radial_profile<NT_MESH, NT_WIND, bin_max>(
            sx, sy, mx, my, wu, wv, nx, ny, nr, nt, dr, dt, r_core, r_max,
            r_all, theta_all, w_all);
    }

    // function generate Python code to plot the radial profile
    // for debuging
    template<typename NT_MESH, typename NT_WIND>
    static void plot_radial_profile(std::ostream &ostr,
        unsigned int k, unsigned long track_id, unsigned int nr,
        unsigned int nt, unsigned int n_crit,
        p_teca_variant_array_impl<NT_MESH> r_all,
        p_teca_variant_array_impl<NT_MESH> t_all,
        p_teca_variant_array_impl<NT_WIND> w_all,
        p_teca_variant_array_impl<NT_MESH> r,
        p_teca_variant_array_impl<NT_MESH> t,
        p_teca_variant_array_impl<NT_WIND> w_prof,
        p_teca_variant_array_impl<NT_MESH> r_crit,
        p_teca_variant_array_impl<NT_MESH> r_peak,
        p_teca_variant_array_impl<NT_WIND> w_peak,
        NT_WIND *w_crit);
};

template<typename NT>
class teca_tc_wind_radii::internals_t::bin_average
{
public:
    bin_average() = delete;
    bin_average(unsigned int nr, unsigned int nt)
        : m_nr(nr), m_nt(nt), m_nrnt(nr*nt)
    {
        m_vals = teca_variant_array_impl<NT>::New(m_nrnt, NT());
        m_pvals = m_vals->get();

        m_count = teca_unsigned_int_array::New(m_nrnt, 0);
        m_pcount = m_count->get();
    }

    void operator()(unsigned int r, unsigned int t, NT val)
    {
        unsigned int q = t*m_nr + r;
        m_pvals[q] += val;
        m_pcount[q] += 1;
    }

    p_teca_variant_array_impl<NT> get_bin_values()
    {
        for (unsigned int i = 0; i < m_nrnt; ++i)
            m_pvals[i] = m_pcount[i] ? m_pvals[i]/m_pcount[i] : m_pvals[i];
        return m_vals;
    }

    bool valid()
    {
        for (unsigned int j = 0; j < m_nt; ++j)
        {
            for (unsigned int i = 0; i < m_nr; ++i)
            {
                unsigned int q = j*m_nr + i;
                if (!m_pcount[q])
                {
                    TECA_ERROR("bin r_" << i << ", theta_" << j << " is empty")
                    return false;
                }
            }
        }
        return true;
    }

private:
    p_teca_variant_array_impl<NT> m_vals;
    NT *m_pvals;
    p_teca_unsigned_int_array m_count;
    unsigned int *m_pcount;
    unsigned int m_nr;
    unsigned int m_nt;
    unsigned int m_nrnt;
};


template<typename NT>
class teca_tc_wind_radii::internals_t::bin_max
{
public:
    bin_max() = delete;
    bin_max(unsigned int nr, unsigned int nt)
        : m_nr(nr), m_nt(nt), m_nrnt(nr*nt)
    {
        m_vals = teca_variant_array_impl<NT>::New(m_nr*m_nt, NT());
        m_pvals = m_vals->get();

        m_count = teca_unsigned_int_array::New(m_nrnt, 0);
        m_pcount = m_count->get();
    }

    void operator()(unsigned int r, unsigned int t, NT val)
    {
        unsigned int q = t*m_nr + r;
        m_pvals[q] = std::max(m_pvals[q], val);
        m_pcount[q] += 1;
    }

    p_teca_variant_array_impl<NT> get_bin_values()
    { return m_vals; }

    bool valid()
    {
        for (unsigned int j = 0; j < m_nt; ++j)
        {
            for (unsigned int i = 0; i < m_nr; ++i)
            {
                unsigned int q = j*m_nr + i;
                if (!m_pcount[q])
                {
                    TECA_ERROR("bin r_" << i << ", theta_" << j << " is empty")
                    return false;
                }
            }
        }
        return true;
    }
private:
    p_teca_variant_array_impl<NT> m_vals;
    NT *m_pvals;
    p_teca_unsigned_int_array m_count;
    unsigned int *m_pcount;
    unsigned int m_nr;
    unsigned int m_nt;
    unsigned int m_nrnt;
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
    this->track_table = nullptr;
    this->number_of_storms = 0;
    this->metadata.clear();
    this->track_counts.clear();
    this->track_offsets.clear();
    this->track_ids.clear();
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_tc_wind_radii::internals_t::copy_expand_array(
    const_p_teca_variant_array va_in, unsigned long n_per)
{
    unsigned long n = va_in->size();
    p_teca_variant_array va_out = va_in->new_instance(n*n_per);
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        va_out.get(),
        const NT *pva_in = static_cast<const TT*>(va_in.get())->get();
        NT *pva_out = static_cast<TT*>(va_out.get())->get();
        for (unsigned long k = 0; k < n; ++k)
        {
            NT va_in_k = pva_in[k];
            NT *pva_out_k = pva_out + k*n_per;
            for (unsigned long j = 0; j < n_per; ++j)
                pva_out_k[j] = va_in_k;
        }
        )
    return va_out;
}


// --------------------------------------------------------------------------
template<typename NT_MESH>
int teca_tc_wind_radii::internals_t::compute_sampling_parameters(
    const NT_MESH *pmx, const NT_MESH *pmy, unsigned long nx, unsigned long ny,
    NT_MESH search_radius, unsigned int &r_resolution, NT_MESH &core_radius)
{
    // inside the core radius we sample all points, core radius
    // is a function of the grid spacing
    NT_MESH r_cell = 0.0;
    if ((core_radius <= 0.0) || (r_resolution == 0))
    {
        NT_MESH mdx = std::numeric_limits<NT_MESH>::lowest();
        unsigned long nxm1 = nx - 1;
        for (unsigned long  i = 1; i < nxm1; ++i)
            mdx = std::max(mdx, (pmx[i+1]-pmx[i-1])/NT_MESH(2.0));
        NT_MESH mdy = std::numeric_limits<NT_MESH>::lowest();
        unsigned long nym1 = ny - 1;
        for (unsigned long  i = 1; i < nym1; ++i)
            mdy = std::max(mdy, std::fabs(pmy[i+1]-pmy[i-1])/NT_MESH(2.0));
        r_cell = std::sqrt(mdx*mdx + mdy*mdy);
    }

    core_radius = core_radius <= NT_MESH() ?  NT_MESH(2)*r_cell : core_radius;

    r_resolution = r_resolution == 0 ? static_cast<unsigned int>
        (search_radius/(NT_MESH(0.5)*r_cell)) : r_resolution;

    return 0;
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
int teca_tc_wind_radii::internals_t::compute_crossings(NT_MESH *r, NT_WIND *w,
    NT_WIND *w_crit, unsigned int nr, unsigned int nt, unsigned int n_crit,
    unsigned int *crit_ids, NT_MESH *r_cross)
{
    for (unsigned int j = 0; j < nt; ++j)
    {
        // for each critical speed where a radial crossing was detected
        // solve for the interecpt of the linear approximtion of the
        // radial profile and the horizontal line definied by the critical
        // wind value
        unsigned int jnc = j*n_crit;
        unsigned int *cc = crit_ids + jnc;
        NT_MESH *rc = r_cross + jnc;

        unsigned int jnr = j*nr;
        NT_WIND *ww = w + jnr;

        for (unsigned int i = 0; i < n_crit; ++i)
        {
            if (cc[i])
            {
                // by construction we know crossing is in between these ids
                // the ids name 2 points defining a line guaranteed to intercept
                // the horizontal line defined by the critical wind speed
                unsigned int q2 = cc[i];
                unsigned int q1 = q2-1;

                compute_crossing<NT_MESH>(r[q1], ww[q1],
                     r[q2], ww[q2], w_crit[i], rc[i]);
            }
        }
    }

    return 0;
}

// --------------------------------------------------------------------------
template <typename NT_MESH, typename NT_WIND>
int teca_tc_wind_radii::internals_t::locate_critical_ids(NT_MESH *r,
    NT_WIND *w, NT_WIND *w_crit, unsigned int nr, unsigned int nt,
    unsigned int n_crit, unsigned int *crit_ids, unsigned int *peak_id)
{
    (void)r;

    for (unsigned int j = 0; j < nt; ++j)
    {
        NT_WIND *ww = w + j*nr;

        // locate the w_peak and r_peak at theta_j
        peak_id[j] = 0;
        for (unsigned int i = 1; i < nr; ++i)
            peak_id[j] = ww[i] > ww[peak_id[j]] ? i : peak_id[j];

        // locate the critical values
        unsigned int *cc = crit_ids + j*n_crit;
        for (unsigned int i = 0; i < n_crit; ++i)
        {
            // skip when search is impossible
            if (w_crit[i] >= ww[peak_id[j]])
                continue;

            // find the first less or equal to the critical value
            // from the peak
            for (unsigned int q = peak_id[j]; (q < nr) && !cc[i]; ++q)
                cc[i] = ww[q] < w_crit[i] ? q : 0;
        }
    }

    return 0;
}

// --------------------------------------------------------------------------
template<typename NT_MESH, typename NT_WIND,
    template<typename> class bin_operation_t>
p_teca_variant_array_impl<NT_WIND>
teca_tc_wind_radii::internals_t::compute_radial_profile(NT_MESH sx,
    NT_MESH sy, const NT_MESH *mx, const NT_MESH *my, const NT_WIND *wu,
    const NT_WIND *wv, unsigned long nx, unsigned long ny, unsigned int nr,
    unsigned int nt, NT_MESH dr, NT_MESH dt, NT_MESH r_core, NT_MESH r_max,
    p_teca_variant_array_impl<NT_MESH> &r_all,
    p_teca_variant_array_impl<NT_MESH> &theta_all,
    p_teca_variant_array_impl<NT_WIND> &w_all)
{
#if defined(TECA_DEBUG)
    r_all = teca_variant_array_impl<NT_MESH>::New();
    theta_all = teca_variant_array_impl<NT_MESH>::New();
    w_all = teca_variant_array_impl<NT_WIND>::New();
#else
    (void)r_all;
    (void)theta_all;
    (void)w_all;
#endif

    // construct an instance of the binning operator
    bin_operation_t<NT_WIND> bin_op(nr, nt);

    // for each grid point compute radial distance to storm center
    for (unsigned long j = 0; j < ny; ++j)
    {
        unsigned long q = j*nx;
        NT_MESH y = my[j] - sy;
        NT_MESH yy = y*y;
        for (unsigned long i = 0; i < nx; ++i)
        {
            // compute r
            NT_MESH x = mx[i] - sx;
            NT_MESH xx = x*x;
            NT_MESH r = std::sqrt(xx + yy);

            // sample wind onto the discrete r, theta mesh
            // using the desired binning operation
            if (r <= r_max)
            {
                // compute wind speed at the grid point
                NT_WIND w = teca_tc_wind_radii::internals_t::
                    compute_wind_speed(x, y, wu[q+i], wv[q+i]);

                // compute theta
                NT_MESH t = std::atan2(x, y);
                t = t < NT_MESH() ? NT_MESH(2.0*M_PI) + t : t;

                unsigned int i_r = static_cast<unsigned int>(r/dr);
                unsigned int i_t = static_cast<unsigned int>(t/dt);

                // with 0.25 degree res and lower, we don't have enough
                // grid points close to storm core so always include what
                // ever we do have in the profile
                if (r <= r_core)
                    for (unsigned int k = 0; k < nt; ++k)
                        bin_op(i_r, k, w);
                else
                    bin_op(i_r, i_t, w);

#if defined(TECA_DEBUG)
                r_all->append(r);
                theta_all->append(t);
                w_all->append(w);
#endif
            }
        }
    }

    if (!bin_op.valid())
        return nullptr;

    return bin_op.get_bin_values();
}

// --------------------------------------------------------------------------
template <typename NT_MESH, typename NT_WIND>
int teca_tc_wind_radii::internals_t::compute_wind_radii(unsigned int k,
    unsigned long track_id, int profile_type, unsigned int nr, unsigned int nt,
    NT_MESH r_core, NT_MESH r_max, NT_MESH sx, NT_MESH sy, const NT_MESH *mx,
    const NT_MESH *my, const NT_WIND *wu, const NT_WIND *wv, unsigned long nx,
    unsigned long ny, NT_WIND *w_crit, unsigned int n_crit,
    p_teca_variant_array_impl<NT_MESH> &r_crit,
    p_teca_variant_array_impl<NT_MESH> &r_peak,
    p_teca_variant_array_impl<NT_WIND> &w_peak)
{
    // construct radial discretization
    NT_MESH dr = r_max/static_cast<NT_MESH>(nr);
    NT_MESH dr2 = dr/NT_MESH(2);

    p_teca_variant_array_impl<NT_MESH> r = teca_variant_array_impl<NT_MESH>::New(nr);
    NT_MESH *pr = r->get();

    for (unsigned int i = 0; i < nr; ++i)
        pr[i] = dr2 + static_cast<NT_MESH>(i)*dr;

    p_teca_variant_array_impl<NT_MESH> r_all;
    p_teca_variant_array_impl<NT_MESH> t_all;
    p_teca_variant_array_impl<NT_WIND> w_all;

    // construct theta discretization
    NT_MESH dt = NT_MESH(2.0*M_PI)/nt;

#if defined(TECA_DEBUG)
    p_teca_variant_array_impl<NT_MESH> theta =
         teca_variant_array_impl<NT_MESH>::New(nt);

    NT_MESH *pt = theta->get();

    for (unsigned int i = 0; i < nt; ++i)
        pt[i] = static_cast<NT_MESH>(i)*dt;
#endif

    // compute the radial profiles
    p_teca_variant_array_impl<NT_WIND> w_prof;
    switch (profile_type)
    {
    case PROFILE_AVERAGE:
        w_prof = teca_tc_wind_radii::internals_t::compute_average_radial_profile
            (sx, sy, mx, my, wu, wv, nx, ny, nr, nt, dr, dt, r_core, r_max,
            r_all, t_all, w_all);
        break;
    case PROFILE_MAX:
        w_prof = teca_tc_wind_radii::internals_t::compute_max_radial_profile
            (sx, sy, mx, my, wu, wv, nx, ny, nr, nt, dr, dt, r_core, r_max,
            r_all, t_all, w_all);
        break;
    default:
        TECA_ERROR("Invalid profile type \"" << profile_type << "\"")
        return -1;
    }
    if (!w_prof)
    {
        TECA_ERROR("Sampling parameters nr=" << nr << " dr=" << dr << " nt=" << nt
            << " dt=" << dt << " resulted in an incomplete wind profile for track "
            << track_id << " storm " << k)
        return -1;
    }

    // compute the offsets of the critical radii
    unsigned int ncnt = n_crit*nt;
    p_teca_unsigned_int_array crit_ids = teca_unsigned_int_array::New(ncnt, 0u);
    unsigned int *pcrit_ids = crit_ids->get();

    p_teca_unsigned_int_array peak_id = teca_unsigned_int_array::New(nt, 0u);
    unsigned int *ppeak_id = peak_id->get();

    NT_WIND *pw = w_prof->get();

    teca_tc_wind_radii::internals_t::locate_critical_ids
        (pr, pw, w_crit, nr, nt, n_crit, pcrit_ids, ppeak_id);

    // compute the intercepts with the critical wind_profile speeds
    r_crit = teca_variant_array_impl<NT_MESH>::New(ncnt, NT_MESH());
    NT_MESH *pr_crit = r_crit->get();

    teca_tc_wind_radii::internals_t::compute_crossings
        (pr, pw, w_crit, nr, nt, n_crit, pcrit_ids, pr_crit);

    // from peak_id look up r_peak and w_peak
    r_peak = teca_variant_array_impl<NT_MESH>::New(nt);
    NT_MESH *pr_peak = r_peak->get();

    for (unsigned int i = 0; i < nt; ++i)
        pr_peak[i] = pr[ppeak_id[i]];

    // from peak_id look up w_peak and w_peak
    w_peak = teca_variant_array_impl<NT_WIND>::New(nt);
    NT_WIND *pw_peak = w_peak->get();

    for (unsigned int i = 0; i < nt; ++i)
        pw_peak[i] = pw[i*nr + ppeak_id[i]];

#if defined(TECA_DEBUG)
    teca_tc_wind_radii::internals_t::plot_radial_profile(cout,
        k, track_id, nr, nt, n_crit, r_all, t_all, w_all, r,
        theta, w_prof, r_crit, r_peak, w_peak, w_crit);
#endif

    return 0;
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
void teca_tc_wind_radii::internals_t::plot_radial_profile(std::ostream &ostr,
    unsigned int k, unsigned long track_id,  unsigned int nr, unsigned int nt,
    unsigned int n_crit, p_teca_variant_array_impl<NT_MESH> r_all,
    p_teca_variant_array_impl<NT_MESH> t_all,
    p_teca_variant_array_impl<NT_WIND> w_all,
    p_teca_variant_array_impl<NT_MESH> r,
    p_teca_variant_array_impl<NT_MESH> t,
    p_teca_variant_array_impl<NT_WIND> w_prof,
    p_teca_variant_array_impl<NT_MESH> r_crit,
    p_teca_variant_array_impl<NT_MESH> r_peak,
    p_teca_variant_array_impl<NT_WIND> w_peak,
    NT_WIND *w_crit)

{
    ostr << "r_all = np.array(["; r_all->to_stream(ostr); ostr << "])" << endl;
    ostr << "t_all = np.array(["; t_all->to_stream(ostr); ostr << "])" << endl;
    ostr << "w_all = np.array(["; w_all->to_stream(ostr); ostr << "])" << endl;
    ostr << "r = np.array(["; r->to_stream(ostr); ostr << "])" << endl;
    ostr << "theta = np.array(["; t->to_stream(ostr); ostr << "])" << endl;
    ostr << "w_prof = np.array(["; w_prof->to_stream(ostr); ostr << "])" << endl;
    ostr << "r_crit = np.array(["; r_crit->to_stream(ostr); ostr << "])" << endl;
    ostr << "r_peak = np.array(["; r_peak->to_stream(ostr); ostr << "])" << endl;
    ostr << "w_peak = np.array(["; w_peak->to_stream(ostr); ostr << "])" << endl;

    ostr << "w_crit = np.array([" << w_crit[0];
    for (unsigned int i = 1; i < n_crit; ++i)
        ostr << ", " << w_crit[i];
    ostr << "])" << endl;

    ostr << "dt = " << "2.0*np.pi/" << nt << endl
        << "dom = [0, max(r_all)]" << endl
        << "rng = [0, 1.1*max(max(w_crit), max(w_peak))]" << endl
        << "fig = mpl.figure(figsize=(5, max(3, " << nt << ")))" << endl;

    for (unsigned int j = 0; j < nt; ++j)
    {
        unsigned int jnr = j*nr;
        unsigned int jnc = j*n_crit;

        ostr << "ax = mpl.subplot(" << nt << ",1," << j+1 << ")" << endl;

        // data
        ostr << "ii = np.where(np.logical_and((t_all >= dt*"
             << j <<"), (t_all < dt*" << j+1 << ")))" << endl;

        ostr << "mpl.plot(r_all[ii], w_all[ii], '.', markerfacecolor='none',"
            << " markeredgecolor='g', alpha=0.5)" << endl;

        // w_prof
        ostr << "mpl.plot(r, w_prof[" << jnr << ":" << jnr+nr << "], 'k-', linewidth=2)" << endl
            << "mpl.plot(r, w_prof[" << jnr << ":" << jnr+nr << "], 'k.')" << endl;

        // w_crit
        for (unsigned int i = 0; i < n_crit; ++i)
            ostr << "mpl.plot(dom, [w_crit[" << i << "]]*2, 'r--', alpha=0.5)" << endl;

        // r_crit
        for (unsigned int i = 0; i < n_crit; ++i)
            ostr << "mpl.plot([r_crit[" << jnc+i << "]]*2, [0, w_crit[" << i << "]],"
               << " 'b--', alpha=0.5)" << endl;

        ostr << "mpl.plot(r_crit[" << jnc << ":" << jnc+n_crit << "], w_crit, 'bo',"
            << " markerfacecolor='y', markeredgewidth=2)" << endl;

        // r_peak
        ostr << "mpl.plot([r_peak[" << j << "]]*2, [0, w_peak[" << j << "]],"
            << " 'b--', alpha=0.5)" << endl;

        ostr << "mpl.plot(r_peak[" << j << "], w_peak[" << j << "], 'b^',"
            << " markerfacecolor='none', markeredgewidth=2)" << endl;

        ostr << "mpl.grid(True)" << endl
            << "mpl.xlim(dom)" << endl
            << "mpl.ylim(rng)" << endl
            << "mpl.setp(ax.get_xticklabels(), visible=False)" << endl;
    }

    // format the plot
    ostr << "mpl.setp(ax.get_xticklabels(), visible=True)" << endl
        << "mpl.suptitle('radial profile track=" << track_id << " step=" << k << "')" << endl
        << "fig.text(0.5, 0.04, 'dist to storm center (deg lat)', ha='center')" << endl
        << "fig.text(0.04, 0.5, 'wind (m/s)', va='center', rotation='vertical')" << endl
        << "mpl.subplots_adjust(hspace=0.1, top=0.95)" << endl;

    // save it
    ostr << "mpl.savefig('radial_wind_profile_"
        << std::setfill('0') << std::setw(5) << track_id << "_"
        << std::setfill('0') << std::setw(5) << k << ".png', dpi=100)"
        << endl;

    ostr << "mpl.close(fig)" << endl
        << "sys.stderr.write('*')" << endl;
}


// --------------------------------------------------------------------------
teca_tc_wind_radii::teca_tc_wind_radii() : track_id_column("track_id"),
    track_x_coordinate_column("lon"), track_y_coordinate_column("lat"),
    track_wind_speed_column("surface_wind"), track_time_column("time"),
    wind_u_variable("UBOT"), wind_v_variable("VBOT"),
    critical_wind_speeds({
        teca_saffir_simpson::get_upper_bound_mps<double>(-1),
        teca_saffir_simpson::get_upper_bound_mps<double>(0),
        teca_saffir_simpson::get_upper_bound_mps<double>(1),
        teca_saffir_simpson::get_upper_bound_mps<double>(2),
        teca_saffir_simpson::get_upper_bound_mps<double>(3),
        teca_saffir_simpson::get_upper_bound_mps<double>(4)}),
    search_radius(6.0), core_radius(0), r_resolution(0),
    theta_resolution(1), profile_type(PROFILE_AVERAGE)
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
        TECA_POPTS_GET(std::string, prefix, track_id_column,
            "name of the column containing unique ids of the tracks")
        TECA_POPTS_GET(std::string, prefix, track_x_coordinate_column,
            "name of the column to create track x coordinates from")
        TECA_POPTS_GET(std::string, prefix, track_y_coordinate_column,
            "name of the column to create track y coordinates from")
        TECA_POPTS_GET(std::string, prefix, track_time_column,
            "name of the column to create track times from")
        TECA_POPTS_GET(std::string, prefix, wind_u_variable,
            "name of the variable containing u component of wind")
        TECA_POPTS_GET(std::string, prefix, wind_v_variable,
            "name of the variable containing v component of wind")
        TECA_POPTS_GET(double, prefix, search_radius,
            "defines the radius of the search space in deg lat")
        TECA_POPTS_MULTI_GET(std::vector<double>, prefix, critical_wind_speeds,
            "sets the wind speeds to compute radii at")
        TECA_POPTS_GET(int, prefix, r_resolution,
            "sets the number of bins to discretize in the radial direction")
        TECA_POPTS_GET(int, prefix, theta_resolution,
            "sets the number of bins to discretize in the theta direction")
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
    TECA_POPTS_SET(opts, std::string, prefix, track_id_column)
    TECA_POPTS_SET(opts, std::string, prefix, track_x_coordinate_column)
    TECA_POPTS_SET(opts, std::string, prefix, track_y_coordinate_column)
    TECA_POPTS_SET(opts, std::string, prefix, track_time_column)
    TECA_POPTS_SET(opts, std::string, prefix, wind_u_variable)
    TECA_POPTS_SET(opts, std::string, prefix, wind_v_variable)
    TECA_POPTS_SET(opts, std::vector<double>, prefix, critical_wind_speeds)
    TECA_POPTS_SET(opts, double, prefix, search_radius)
    TECA_POPTS_SET(opts, double, prefix, core_radius)
    TECA_POPTS_SET(opts, int, prefix, r_resolution)
    TECA_POPTS_SET(opts, int, prefix, theta_resolution)
    TECA_POPTS_SET(opts, int, prefix, profile_type)
}
#endif

// --------------------------------------------------------------------------
void teca_tc_wind_radii::set_input_connection(unsigned int id,
        const teca_algorithm_output_port &port)
{
    if (id == 0)
        this->internals->track_pipeline_port = port;
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

    if (this->internals->track_table)
        return this->internals->metadata;

    // execute the pipeline that retruns table of tracks
    const_p_teca_dataset track_data;

    p_teca_programmable_algorithm capture_track_data
        = teca_programmable_algorithm::New();

    capture_track_data->set_input_connection(this->internals->track_pipeline_port);

    capture_track_data->set_execute_callback(
        [&track_data] (unsigned int, const std::vector<const_p_teca_dataset> &in_data,
     const teca_metadata &) -> const_p_teca_dataset
     {
         track_data = in_data[0];
         return nullptr;
     });

    capture_track_data->update();

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
        const_p_teca_table track_table =
            std::dynamic_pointer_cast<const teca_table>(track_data);

        if (!track_table)
        {
            TECA_ERROR("metadata pipeline failure")
        }

        // column need to build random access data structures
        const_p_teca_variant_array track_ids =
            track_table->get_column(this->track_id_column);

        if (!track_ids)
        {
            TECA_ERROR("track index column \""
            << this->track_id_column << "\" not found")
        }
        // these columns are needed to compute the track size
        else
        if (!track_table->has_column(this->track_x_coordinate_column))
        {
            TECA_ERROR("track x coordinates column \""
                << this->track_x_coordinate_column << "\" not found")
        }
        else
        if (!track_table->has_column(this->track_y_coordinate_column))
        {
            TECA_ERROR("track y coordinates column \""
                << this->track_y_coordinate_column << "\" not found")
        }
        else
        if (!track_table->has_column(this->track_wind_speed_column))
        {
            TECA_ERROR("track wind speed column \""
                << this->track_wind_speed_column << "\" not found")
        }
        else
        if (!track_table->has_column(this->track_time_column))
        {
            TECA_ERROR("track time column \""
                << this->track_time_column << "\" not found")
        }
        // things are ok, take a reference
        else
        {
            this->internals->track_table = track_table;
        }
    }

    // distribute the table to all processes
#if defined(TECA_HAS_MPI)
    if (is_init)
    {
        teca_binary_stream bs;
        if (this->internals->track_table && (rank == 0))
            this->internals->track_table->to_stream(bs);
        bs.broadcast();
        if (bs && (rank != 0))
        {
           p_teca_table tmp = teca_table::New();
           tmp->from_stream(bs);
           this->internals->track_table = tmp;
        }
    }
#endif

    // build random access data structures
    const_p_teca_variant_array track_ids =
        this->internals->track_table->get_column(this->track_id_column);

    TEMPLATE_DISPATCH_I(const teca_variant_array_impl,
        track_ids.get(),

        const NT *ptrack_ids = dynamic_cast<TT*>(track_ids.get())->get();

        teca_coordinate_util::get_table_offsets(ptrack_ids,
            this->internals->track_table->get_number_of_rows(),
            this->internals->number_of_storms, this->internals->track_counts,
            this->internals->track_offsets, this->internals->track_ids);
        )

    // must have at least one time storm
    if (this->internals->number_of_storms < 1)
    {
        TECA_ERROR("Invalid index \"" << this->track_id_column << "\"")
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
    unsigned long id_ofs = this->internals->track_offsets[map_id];
    unsigned long n_ids = this->internals->track_counts[map_id];

    const_p_teca_variant_array
    x_coordinates = this->internals->track_table->get_column
            (this->track_x_coordinate_column);

    const_p_teca_variant_array
    y_coordinates = this->internals->track_table->get_column
            (this->track_y_coordinate_column);

    const_p_teca_variant_array
    times = this->internals->track_table->get_column
            (this->track_time_column);

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
            // TODO -- Haversine
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
    cout << "import matplotlib.pyplot as mpl, numpy as np, sys" << endl;
#endif
    (void)port;

    // check for empty input
    const_p_teca_cartesian_mesh mesh
         = std::dynamic_pointer_cast<const teca_cartesian_mesh>
            (input_data.size() ? input_data[0] : nullptr);

    if (!mesh)
    {
        TECA_ERROR("Invalid input mesh dataset.")
        return nullptr;
    }

    // get id of storm id being requested
    unsigned long map_id = 0;
    request.get("time_step", map_id);

    // for random access into the specific track
    unsigned long ofs = this->internals->track_offsets[map_id];
    unsigned long npts = this->internals->track_counts[map_id];

    // get strom track positions
    const_p_teca_variant_array track_x =
        this->internals->track_table->get_column(this->track_x_coordinate_column);

    const_p_teca_variant_array track_y =
        this->internals->track_table->get_column(this->track_y_coordinate_column);

    const_p_teca_variant_array track_t =
        this->internals->track_table->get_column(this->track_time_column);

    const_p_teca_variant_array track_ids =
        this->internals->track_table->get_column(this->track_id_column);

    // allocate output columns
    // each point in the track gets a set of curves r_ij = f(theta_j)
    unsigned long n_rows = npts*this->theta_resolution;

    // pass track id on, this will be used to cross reference back
    // in to the tracks table
    unsigned long orig_track_id = 0;
    track_ids->get(ofs, orig_track_id);

    p_teca_unsigned_long_array track_id =
        teca_unsigned_long_array::New(n_rows, orig_track_id);

    // j identifies entries at the given instant along the track
    p_teca_unsigned_long_array track_index =
        teca_unsigned_long_array::New(n_rows);
    for (unsigned long k = 0; k < npts; ++k)
    {
        unsigned long kk = k*this->theta_resolution;
        unsigned long *pidx = track_index->get() + kk;
        for (unsigned int j = 0; j < this->theta_resolution; ++j)
            pidx[j] = k;
    }

    // track time
    p_teca_variant_array time = internals_t::copy_expand_array
            (track_t, this->theta_resolution);

    // track points
    p_teca_variant_array x = internals_t::copy_expand_array
            (track_x, this->theta_resolution);

    p_teca_variant_array y = internals_t::copy_expand_array
            (track_y, this->theta_resolution);

    // theta_j
    p_teca_variant_array theta =
        mesh->get_x_coordinates()->new_instance(n_rows);

    TEMPLATE_DISPATCH(teca_variant_array_impl,
        theta.get(),
        for (unsigned long k = 0; k < npts; ++k)
        {
            unsigned long kk = k*this->theta_resolution;
            NT dt = 2.0f*float(M_PI)/float(this->theta_resolution);
            NT dt2 = dt/2.0;
            NT *pt = static_cast<TT*>(theta.get())->get() + kk;
            for (unsigned int j = 0; j < this->theta_resolution; ++j)
                pt[j] = dt2 + j*dt;
        }
        )

    // r_ij = R0j ... Rnj
    unsigned int n_crit = this->critical_wind_speeds.size();
    std::vector<p_teca_variant_array> r_crit(n_crit);
    for (unsigned int i = 0; i < n_crit; ++i)
        r_crit[i] = mesh->get_x_coordinates()->new_instance(n_rows);

    // RP, WP
    p_teca_variant_array r_peak =
        mesh->get_x_coordinates()->new_instance(n_rows);

    p_teca_variant_array w_peak = mesh->get_point_arrays()->get
        (this->wind_u_variable)->new_instance(n_rows);

    // compute radius at each point in time along the storm track
    NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
        track_x.get(), _TRACK,

        // get the storm centers
        const NT_TRACK *ptrack_x = static_cast<TT_TRACK*>(track_x.get())->get();
        const NT_TRACK *ptrack_y = static_cast<TT_TRACK*>(track_y.get())->get();

        // for each time instance in the storm compute the wind radii
        for (unsigned long k = 0; k < npts; ++k)
        {
            unsigned long kk = k*this->theta_resolution;

            // get the kth mesh
            mesh = std::dynamic_pointer_cast<const teca_cartesian_mesh>(input_data[k]);
            if (!mesh)
            {
                TECA_ERROR("input " << k << " is empty or not a cartesian mesh")
                return nullptr;
            }

            // mesh coords
            const_p_teca_variant_array mesh_x = mesh->get_x_coordinates();
            const_p_teca_variant_array mesh_y = mesh->get_y_coordinates();

            unsigned long nx = mesh_x->size();
            unsigned long ny = mesh_y->size();

            // wind components
            const_p_teca_variant_array wind_u =
                mesh->get_point_arrays()->get(this->wind_u_variable);

            const_p_teca_variant_array wind_v =
                mesh->get_point_arrays()->get(this->wind_v_variable);

            NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
                mesh_x.get(), _MESH,
                using TT_MESH_OUT = teca_variant_array_impl<NT_MESH>;
                using P_TT_MESH_OUT = std::shared_ptr<TT_MESH_OUT>;

                NESTED_TEMPLATE_DISPATCH_FP(const teca_variant_array_impl,
                    wind_u.get(), _WIND,
                    using TT_WIND_OUT = teca_variant_array_impl<NT_WIND>;
                    using P_TT_WIND_OUT = std::shared_ptr<TT_WIND_OUT>;

                    // mesh coords
                    const NT_MESH *pmx = static_cast<TT_MESH*>(mesh_x.get())->get();
                    const NT_MESH *pmy = static_cast<TT_MESH*>(mesh_y.get())->get();

                    // wind
                    const NT_WIND *pwu = static_cast<TT_WIND*>(wind_u.get())->get();
                    const NT_WIND *pwv = static_cast<TT_WIND*>(wind_v.get())->get();

                    // storm center
                    NT_MESH sx = static_cast<NT_MESH>(ptrack_x[k+ofs]);
                    NT_MESH sy = static_cast<NT_MESH>(ptrack_y[k+ofs]);

                    // params
                    NT_MESH r_max = static_cast<NT_MESH>(this->search_radius);
                    NT_MESH r_core = static_cast<NT_MESH>(this->core_radius);
                    unsigned int r_res = this->r_resolution;

                    internals_t::compute_sampling_parameters(pmx, pmy, nx, ny,
                        r_max, r_res, r_core);

                    std::vector<NT_WIND> w_crit(this->critical_wind_speeds.begin(),
                        this->critical_wind_speeds.end());

                    // results
                    P_TT_MESH_OUT r_crit_k;
                    P_TT_MESH_OUT r_peak_k;
                    P_TT_WIND_OUT w_peak_k;

                    // compute the wind radii
                    if (internals_t::compute_wind_radii(k, orig_track_id, this->profile_type,
                        r_res, this->theta_resolution, r_core, r_max, sx, sy, pmx, pmy, pwu,
                        pwv, nx, ny, w_crit.data(), n_crit, r_crit_k, r_peak_k, w_peak_k))
                    {
                        TECA_ERROR("Failed to compute radial profiles for track "
                            << orig_track_id << " storm " << k)
                        return nullptr;
                    }

                    // copy into the output columns, ri need to be moved from
                    // a 2d array into columnar arrays.
                    NT_MESH *prck = r_crit_k->get();
                    for (unsigned int i = 0; i < n_crit; ++i)
                    {
                        NT_MESH *prcoi = static_cast<TT_MESH_OUT*>
                            (r_crit[i].get())->get() + kk;

                        for (unsigned int j = 0; j < this->theta_resolution; ++j)
                            prcoi[j] = prck[j*n_crit+i];
                    }

                    NT_MESH *rp = r_peak_k->get();
                    NT_MESH *prpo = static_cast<TT_MESH_OUT*>(r_peak.get())->get() + kk;
                    for (unsigned int j = 0; j < this->theta_resolution; ++j)
                        prpo[j] = rp[j];

                    NT_WIND *wp = w_peak_k->get();
                    NT_WIND *pwpo = static_cast<TT_WIND_OUT*>(w_peak.get())->get() + kk;
                    for (unsigned int j = 0; j < this->theta_resolution; ++j)
                        pwpo[j] = wp[j];

                    )
                )
        }
        )

    // package the results
    p_teca_table output = teca_table::New();
    output->copy_metadata(this->internals->track_table);
    output->get_metadata().insert("core_radius", this->core_radius);
    output->get_metadata().insert("r_resolution", this->r_resolution);
    output->get_metadata().insert("theta_resolution",
        this->theta_resolution);
    output->get_metadata().insert("critical_wind_speeds",
        this->critical_wind_speeds);
    output->append_column(this->track_id_column, track_id);
    output->append_column("track_point_id", track_index);
    output->append_column(this->track_time_column, time);
    output->append_column(this->track_x_coordinate_column, x);
    output->append_column(this->track_y_coordinate_column, y);
    output->append_column("theta", theta);
    for (unsigned int i = 0; i < n_crit; ++i)
    {
        std::ostringstream oss;
        oss << "r_" << i;
        output->append_column(oss.str(), r_crit[i]);
    }
    output->append_column("r_peak", r_peak);
    output->append_column("w_peak", w_peak);

    return output;
}
