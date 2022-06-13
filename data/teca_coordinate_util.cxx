#include "teca_coordinate_util.h"

#include "teca_common.h"
#if defined(TECA_HAS_UDUNITS)
#include "teca_calcalcs.h"
#endif

#include <string>
#include <cstdio>
#include <ctime>
#include <sstream>
#include <iomanip>

namespace teca_coordinate_util
{

// **************************************************************************
bool equal(const const_p_teca_variant_array &array1,
    const const_p_teca_variant_array &array2,
    double aAbsTol, double aRelTol, int &errorNo,
    std::string &errorStr)
{
    std::ostringstream oss;

    errorNo = equal_error::no_error;
    errorStr = "";

    // Arrays of different sizes are different.
    size_t n_elem = array1->size();
    if (n_elem != array2->size())
    {
        oss << "The arrays have different sizes "
            << n_elem << " and " << array2->size();

        errorStr = oss.str();
        errorNo = equal_error::length_missmatch;

        return false;
    }

    // handle POD arrays
    TEMPLATE_DISPATCH(const teca_variant_array_impl,
        array1.get(),

        auto a1 = static_cast<TT*>(array1.get());

        // we know the type of array 1 now, check the type of array 2
        auto a2 = dynamic_cast<TT*>(array2.get());
        if (!a2)
        {
            oss << "The arrays have different types : "
                << array1->get_class_name() << " and "
                << array2->get_class_name();

            errorStr = oss.str();
            errorNo = equal_error::type_missmatch;

            return false;
        }

        // compare elements
        auto a1a = a1->get_cpu_accessible();
        auto a2a = a2->get_cpu_accessible();

        auto pa1 = a1a.get();
        auto pa2 = a2a.get();

        std::string diagnostic;
        for (size_t i = 0; i < n_elem; ++i)
        {
            if (std::isinf(pa1[i]) && std::isinf(pa2[i]))
            {
                // the GFDL TC tracker returns inf for some fields in some cases.
                // warn about it so that it may be addressed in other algorithms.
                oss << "Inf detected in element " << i;

                errorStr = oss.str();
                errorNo = equal_error::invalid_value;
            }
            else if (std::isnan(pa1[i]) || std::isnan(pa2[i]))
            {
                // for the time being, don't allow NaN.
                oss << "NaN detected in element " << i;

                errorStr = oss.str();
                errorNo = equal_error::invalid_value;

                return false;
            }
            else if (!teca_coordinate_util::equal<double>(pa1[i], pa2[i],
                diagnostic, aRelTol, aAbsTol))
            {
                oss << "difference above the prescribed tolerance detected"
                       " in element " << i << ". " << diagnostic;

                errorStr = oss.str();
                errorNo = equal_error::value_missmatch;

                return false;
            }
        }

        // we are here, arrays are the same
        errorStr = "The arrays are equal";
        errorNo = equal_error::no_error;

        return true;
        )
    // handle arrays of strings
    TEMPLATE_DISPATCH_CASE(
        const teca_variant_array_impl, std::string,
        array1.get(),
        if (dynamic_cast<const TT*>(array2.get()))
        {
            auto a1 = static_cast<TT*>(array1.get())->get_cpu_accessible();
            auto a2 = static_cast<TT*>(array2.get())->get_cpu_accessible();

            auto pa1 = a1.get();
            auto pa2 = a2.get();

            for (size_t i = 0; i < n_elem; ++i)
            {
                // compare elements
                const std::string &v1 = pa1[i];
                const std::string &v2 = pa2[i];
                if (v1 != v2)
                {
                    oss << "string element " << i << " not equal. ref value \""
                        << v1 << "\" is not equal to test value \"" << v2 << "\"";

                    errorStr = oss.str();
                    errorNo = equal_error::value_missmatch;

                    return false;
                }
            }

            // we are here, arrays are the same
            errorStr = "The arrays are equal";
            errorNo = equal_error::no_error;

            return true;
        }
        )

    // we are here, array type is not handled
    oss << "The array type " << array1->get_class_name()
        << " is not supported.";

    errorStr = oss.str();
    errorNo = equal_error::unsupported_type;

    return false;
}

// **************************************************************************
int time_step_of(const const_p_teca_variant_array &time,
    bool lower, bool clamp, const std::string &calendar,
    const std::string &units, const std::string &date,
    unsigned long &step)
{
#if defined(TECA_HAS_UDUNITS)
    step = 0;

    // extraxt the time values from the input string
    double s = 0;
    int Y = 0, M = 0, D = 0, h = 0, m = 0;
    int n_conv = sscanf(date.c_str(),
        "%d%*[/-]%d%*[/-]%d %d:%d:%lf", &Y, &M, &D, &h, &m, &s);
    if (n_conv < 1)
    {
        TECA_ERROR("invalid start date \"" << date
            << "\". Date must be in \"YYYY-MM-DD hh:mm:ss\" format")
        return -1;
    }

    // apply calendaring to get a time offset
    double t = 0.0;
    if (teca_calcalcs::coordinate(Y, M, D, h, m, s,
        units.c_str(), calendar.c_str(), &t))
    {
        TECA_ERROR("failed to convert date \"" << date
            << "\" to relative time in the \"" << calendar
            << "\" calendar in units of \"" << units << "\".")

        return -1;
    }

    // locate the nearest time value in the time axis
    unsigned long last = time->size() - 1;
    TEMPLATE_DISPATCH_FP_SI(const teca_variant_array_impl,
        time.get(),

        auto ta = std::static_pointer_cast<TT>(time);
        auto pta = ta->get_cpu_accessible();
        auto p_time = pta.get();

        if (clamp && (t <= p_time[0]))
        {
            step = 0;
        }
        else if (clamp && (t >= p_time[last]))
        {
            step = last;
        }
        else if (teca_coordinate_util::index_of(p_time, 0, last, NT(t), lower, step))
        {
            TECA_ERROR("failed to locate the requested time " << t << " in ["
                << p_time[0] << ", " << p_time[last] << "]")
            return -1;
        }

        return 0;
        )

    TECA_ERROR("time_step_of failed because the internal storage "
        "type used for the time axis (" << time->get_class_name()
        << ") is currently not supported")
    return -1;
#else
    (void)time;
    (void)lower;
    (void)clamp;
    (void)calendar;
    (void)units;
    (void)date;
    step = 0;
    TECA_ERROR("The UDUnits package is required for this operation")
    return -1;
#endif
}


// **************************************************************************
int time_to_string(double val, const std::string &calendar,
  const std::string &units, const std::string &format, std::string &date)
{
#if defined(TECA_HAS_UDUNITS)
    // use teca_calcalcs to convert val to a set of year/month/day/etc.
    struct tm timedata = {};
    double seconds = 0.0;
    if (teca_calcalcs::date(val, &timedata.tm_year, &timedata.tm_mon,
        &timedata.tm_mday, &timedata.tm_hour, &timedata.tm_min, &seconds,
        units.c_str(), calendar.c_str()))
    {
        TECA_ERROR("failed to convert relative time value \"" << val
            << "\" to with the calendar \"" << calendar
            << "\" and units \"" << units << "\".")
        return -1;
    }

    // fix some of the timedata info to conform to what strftime expects
    timedata.tm_year -= 1900; // make the year relative to 1900
    timedata.tm_mon -= 1; // make the months start at 0 instead of 1
    timedata.tm_sec = (int) std::round(seconds);

    // convert the time data to a string
    char tmp[256];
    if (strftime(tmp, sizeof(tmp), format.c_str(), &timedata) == 0)
    {
        TECA_ERROR("failed to convert the time as a string with \""
            << format << "\"")
        return -1;
    }

    date = tmp;

    return 0;
#else
    (void)val;
    (void)calendar;
    (void)units;
    (void)format;
    (void)date;
    TECA_ERROR("The UDUnits package is required for this operation")
    return -1;
#endif
}

// **************************************************************************
int bounds_to_extent(const double *bounds, const teca_metadata &md,
    unsigned long *extent)
{
    teca_metadata coords;
    if (md.get("coordinates", coords))
    {
        TECA_ERROR("Metadata issue, missing cooridnates")
        return -1;
    }

    const_p_teca_variant_array x = coords.get("x");
    const_p_teca_variant_array y = coords.get("y");
    const_p_teca_variant_array z = coords.get("z");

    if (!x || !y || !z)
    {
        TECA_ERROR("Metadata issue, empty coordinate axes")
        return -1;
    }

    if (bounds_to_extent(bounds, x, y, z, extent) ||
        validate_extent(x->size(), y->size(), z->size(), extent, true))
    {
        TECA_ERROR("Invalid bounds raequested [" << bounds[0] << ", "
            << bounds[1] << ", " <<  bounds[2] << ", " << bounds[3] << ", "
            << bounds[4] << ", " << bounds[5] << "]")
        return -1;
    }

    return 0;
}

// **************************************************************************
int bounds_to_extent(const double *bounds,
    const const_p_teca_variant_array &x, const const_p_teca_variant_array &y,
    const const_p_teca_variant_array &z, unsigned long *extent)
{
    TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        x.get(),

        // in the following, for each side (low, high) of the bounds in
        // each cooridnate direction we are searching for the index that
        // is either just below, just above, or exactly at the given value.
        // special cases include:
        //   * x,y,z in descending order. we check for that and
        //     invert the compare functions that define the bracket
        //   * bounds describing a plane. we test for this and
        //     so that both high and low extent return the same value.
        //   * x,y,z are length 1. we can skip the search in that
        //     case.

        const NT eps8 = NT(8)*std::numeric_limits<NT>::epsilon();

        unsigned long nx = x->size();
        unsigned long high_i = nx - 1;
        extent[0] = 0;
        extent[1] = high_i;
        auto xa = std::static_pointer_cast<TT>(x);
        auto pxa = xa->get_cpu_accessible();
        const NT *px = pxa.get();
        NT low_x = static_cast<NT>(bounds[0]);
        NT high_x = static_cast<NT>(bounds[1]);
        bool slice_x = equal(low_x, high_x, eps8);

        unsigned long ny = y->size();
        unsigned long high_j = ny - 1;
        extent[2] = 0;
        extent[3] = high_j;
        auto ya = std::dynamic_pointer_cast<TT>(y);
        auto pya = ya->get_cpu_accessible();
        const NT *py = pya.get();
        NT low_y = static_cast<NT>(bounds[2]);
        NT high_y = static_cast<NT>(bounds[3]);
        bool slice_y = equal(low_y, high_y, eps8);

        unsigned long nz = z->size();
        unsigned long high_k = nz - 1;
        extent[4] = 0;
        extent[5] = high_k;
        auto za = std::dynamic_pointer_cast<TT>(z);
        auto pza = za->get_cpu_accessible();
        const NT *pz = pza.get();
        NT low_z = static_cast<NT>(bounds[4]);
        NT high_z = static_cast<NT>(bounds[5]);
        bool slice_z = equal(low_z, high_z, eps8);

        if (((nx > 1) && (((px[high_i] > px[0]) &&
            (teca_coordinate_util::index_of(px, 0, high_i, low_x, true, extent[0])
            || teca_coordinate_util::index_of(px, 0, high_i, high_x, slice_x, extent[1]))) ||
            ((px[high_i] < px[0]) &&
            (teca_coordinate_util::index_of<NT,descend_bracket<NT>>(px, 0, high_i, low_x, false, extent[0])
            || teca_coordinate_util::index_of<NT,descend_bracket<NT>>(px, 0, high_i, high_x, !slice_x, extent[1])))))

            || ((ny > 1) && (((py[high_j] > py[0]) &&
            (teca_coordinate_util::index_of(py, 0, high_j, low_y, true, extent[2])
            || teca_coordinate_util::index_of(py, 0, high_j, high_y, slice_y, extent[3]))) ||
            ((py[high_j] < py[0]) &&
            (teca_coordinate_util::index_of<NT,descend_bracket<NT>>(py, 0, high_j, low_y, false, extent[2])
            || teca_coordinate_util::index_of<NT,descend_bracket<NT>>(py, 0, high_j, high_y, !slice_y, extent[3])))))

            || ((nz > 1) && (((pz[high_k] > pz[0]) &&
            (teca_coordinate_util::index_of(pz, 0, high_k, low_z, true, extent[4])
            || teca_coordinate_util::index_of(pz, 0, high_k, high_z, slice_z, extent[5]))) ||
            ((pz[high_k] < pz[0]) &&
            (teca_coordinate_util::index_of<NT,descend_bracket<NT>>(pz, 0, high_k, low_z, false, extent[4])
            || teca_coordinate_util::index_of<NT,descend_bracket<NT>>(pz, 0, high_k, high_z, !slice_z, extent[5]))))))

        {
            TECA_ERROR(<< "requested subset [" << bounds[0] << ", " << bounds[1] << ", "
                << bounds[2] << ", " << bounds[3] << ", " << bounds[4] << ", " << bounds[5]
                << "] is not contained in the current dataset bounds ["
                << px[0] << ", " << px[high_i] << ", " << py[0] << ", " << py[high_j] << ", "
                << pz[0] << ", " << pz[high_k] << "]")
            return -1;
        }
        return 0;
        )

    TECA_ERROR("invalid coordinate array type \"" << x->get_class_name() << "\"")
    return -1;
}

// **************************************************************************
int bounds_to_extent(const double *bounds,
    const const_p_teca_variant_array &x, unsigned long *extent)
{
    TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        x.get(),

        // in the following, for each side (low, high) of the bounds in
        // each cooridnate direction we are searching for the index that
        // is either just below, just above, or exactly at the given value.
        // special cases include:
        //   * x,y,z in descending order. we check for that and
        //     invert the compare functions that define the bracket
        //   * bounds describing a plane. we test for this and
        //     so that both high and low extent return the same value.
        //   * x,y,z are length 1. we can skip the search in that
        //     case.

        const NT eps8 = NT(8)*std::numeric_limits<NT>::epsilon();

        unsigned long nx = x->size();
        unsigned long high_i = nx - 1;
        extent[0] = 0;
        extent[1] = high_i;
        auto xa = std::static_pointer_cast<TT>(x);
        auto pxa = xa->get_cpu_accessible();
        const NT *px = pxa.get();
        NT low_x = static_cast<NT>(bounds[0]);
        NT high_x = static_cast<NT>(bounds[1]);
        bool slice_x = equal(low_x, high_x, eps8);

        if (((nx > 1) && (((px[high_i] > px[0]) &&
            (teca_coordinate_util::index_of(px, 0, high_i, low_x, true, extent[0])
            || teca_coordinate_util::index_of(px, 0, high_i, high_x, slice_x, extent[1]))) ||
            ((px[high_i] < px[0]) &&
            (teca_coordinate_util::index_of<NT,descend_bracket<NT>>(px, 0, high_i, low_x, false, extent[0])
            || teca_coordinate_util::index_of<NT,descend_bracket<NT>>(px, 0, high_i, high_x, !slice_x, extent[1]))))))
        {
            TECA_ERROR(<< "requested subset [" << bounds[0] << ", " << bounds[1] << ", "
                << "] is not contained in the current dataset bounds [" << px[0] << ", "
                << px[high_i] << "]")
            return -1;
        }
        return 0;
        )

    TECA_ERROR("invalid coordinate array type \"" << x->get_class_name() << "\"")
    return -1;
}

// **************************************************************************
int validate_centering(int centering)
{
    int ret = -1;
    switch (centering)
    {
        case teca_array_attributes::invalid_value:
            TECA_ERROR("detected invalid_value in centering")
            break;
        case teca_array_attributes::cell_centering:
            ret = 0;
            break;
        case teca_array_attributes::x_face_centering:
            ret = 0;
            break;
        case teca_array_attributes::y_face_centering:
            ret = 0;
            break;
        case teca_array_attributes::z_face_centering:
            ret = 0;
            break;
        case teca_array_attributes::x_edge_centering:
            ret = 0;
            break;
        case teca_array_attributes::y_edge_centering:
            ret = 0;
            break;
        case teca_array_attributes::z_edge_centering:
            ret = 0;
            break;
        case teca_array_attributes::point_centering:
            ret = 0;
            break;
        case teca_array_attributes::no_centering:
            ret = 0;
            break;
        default:
            TECA_ERROR("this centering is undefined " << centering)
    }
    return ret;
}

// **************************************************************************
int get_cartesian_mesh_bounds(const const_p_teca_variant_array x,
    const const_p_teca_variant_array y, const const_p_teca_variant_array z,
    double *bounds)
{
    unsigned long x1 = x->size() - 1;
    unsigned long y1 = y->size() - 1;
    unsigned long z1 = z->size() - 1;

    x->get(0, bounds[0]);
    x->get(x1, bounds[1]);
    y->get(0, bounds[2]);
    y->get(y1, bounds[3]);
    z->get(0, bounds[4]);
    z->get(z1, bounds[5]);

    return 0;
}

// **************************************************************************
int get_cartesian_mesh_extent(const teca_metadata &md,
    unsigned long *whole_extent, double *bounds)
{
    // get the whole extent
    if (md.get("whole_extent", whole_extent, 6))
    {
        TECA_ERROR("metadata is missing \"whole_extent\"")
        return -1;
    }

    if (md.get("bounds", bounds, 6))
    {
        // get coordinates
        teca_metadata coords;
        if (md.get("coordinates", coords))
        {
            TECA_ERROR("metadata is missing \"coordinates\"")
            return -1;
        }

        // get the coordinate arrays
        p_teca_variant_array x, y, z;
        if (!(x = coords.get("x")) || !(y = coords.get("y"))
            || !(z = coords.get("z")))
        {
            TECA_ERROR("coordinate metadata is missing x,y, and or z "
                "coordinate arrays")
            return -1;
        }

        // get bounds of the whole_extent being read
        x->get(whole_extent[0], bounds[0]);
        x->get(whole_extent[1], bounds[1]);
        y->get(whole_extent[2], bounds[2]);
        y->get(whole_extent[3], bounds[3]);
        z->get(whole_extent[4], bounds[4]);
        z->get(whole_extent[5], bounds[5]);
    }

    return 0;
}

// **************************************************************************
int validate_extent(unsigned long nx_max, unsigned long ny_max,
    unsigned long nz_max, unsigned long *extent, bool verbose)
{
    // validate x
    if ((extent[1] >= nx_max) || (extent[1] < extent[0]))
    {
        if (verbose)
        {
            TECA_ERROR("The x-axis extent [" << extent[0] << ", "
                << extent[1] << "] is invalid, the x-axis coordinate"
                " array has " << nx_max << " values")
        }
        return -1;
    }

    // validate y
    if ((extent[3] >= ny_max) || (extent[3] < extent[2]))
    {
        if (verbose)
        {
            TECA_ERROR("The y-axis extent [" << extent[2] << ", "
                << extent[3] << "] is invalid, the y-axis coordinate"
                " array has " << ny_max << " values")
        }
        return -1;
    }

    // validate z
    if ((extent[5] >= nz_max) || (extent[5] < extent[4]))
    {
        if (verbose)
        {
            TECA_ERROR("The z-axis extent [" << extent[4] << ", "
                << extent[5] << "] is invalid, the z-axis coordinate"
                " array has " << nz_max << " values")
        }
        return -1;
    }

    return 0;
}

// **************************************************************************
int clamp_dimensions_of_one(unsigned long nx_max, unsigned long ny_max,
    unsigned long nz_max, unsigned long *extent, bool verbose)
{
    int clamped = 0;

    // clamp x
    if ((nx_max == 1) && (extent[1] != 0))
    {
        if (verbose)
        {
            TECA_WARNING("The requested x-axis extent [" << extent[0] << ", "
                << extent[1] << "] is invalid and was clamped to [0, 0]")
        }
        extent[0] = 0;
        extent[1] = 0;
        clamped = 1;
    }

    // clamp y
    if ((ny_max == 1) && (extent[3] != 0))
    {
        if (verbose)
        {
            TECA_WARNING("The requested y-axis extent [" << extent[2] << ", "
                << extent[3] << "] is invalid and was clamped to [0, 0]")
        }
        extent[2] = 0;
        extent[3] = 0;
        clamped = 1;
    }

    // clamp z
    if ((nz_max == 1) && (extent[5] != 0))
    {
        if (verbose)
        {
            TECA_WARNING("The requested z-axis extent [" << extent[4] << ", "
                << extent[5] << "] is invalid and was clamped to [0, 0]")
        }
        extent[4] = 0;
        extent[5] = 0;
        clamped = 1;
    }

    return clamped;
}


// --------------------------------------------------------------------------
void teca_validate_arrays::set_reference_array(const std::string &a_source,
    const std::string &a_name, const std::string a_units,
    const const_p_teca_variant_array &a_array)
{
    m_reference = a_array;
    m_reference_source = a_source;
    m_reference_name = a_name;
    m_reference_units = a_units;
}

// --------------------------------------------------------------------------
void teca_validate_arrays::append_array(const std::string &a_source,
    const std::string &a_name, const std::string &a_units,
    const const_p_teca_variant_array &a_array)
{
    m_arrays.push_back(a_array);
    m_array_sources.push_back(a_source);
    m_array_names.push_back(a_name);
    m_array_units.push_back(a_units);
}

// --------------------------------------------------------------------------
int teca_validate_arrays::validate(const std::string &a_descriptor,
    double a_abs_tol, double a_rel_tol, std::string &errorStr)
{
    std::ostringstream oss;

    size_t n_arrays = m_arrays.size();
    for (size_t i = 0; i < n_arrays; ++i)
    {
        // check values
        int eNo = 0;
        std::string eStr;

        if (!teca_coordinate_util::equal(m_reference,
            m_arrays[i], a_abs_tol, a_rel_tol, eNo, eStr))
        {
            oss << "The " << a_descriptor << " array \"" << m_array_names[i]
                << "\" from source \"" << m_array_sources[i]
                << "\" does not match the reference " << a_descriptor
                << " array \"" << m_reference_name << "\" from source \""
                << m_reference_source << "\". " << eStr;

            errorStr = oss.str();

            return eNo;
        }

        // check names
        /*if (m_reference_name != m_array_names[i])
        {
            oss << "The " << a_descriptor << " array name " << m_array_names[i]
                << " from source " << m_array_sources[i]
                << " does not match the reference array name " << m_reference_name
                << " from source " << m_reference_source;

            errorStr = oss.str();

            return = -1;
        }*/

        // check units
        if (m_reference_units != m_array_units[i])
        {
            oss << "The " << a_descriptor << " array \"" << m_array_names[i]
                << "\" from source \"" << m_array_sources[i]
                << "\" units \"" << m_array_units[i]
                << "\" does not match the reference " << a_descriptor
                << " array \"" << m_reference_name << "\" from source \""
                << m_reference_source << "\" units \""
                << m_reference_units << "\"";

            errorStr = oss.str();

            return teca_validate_arrays::units_missmatch;
        }
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_coordinate_axis_validator::add_time_axis(const std::string &source,
    const teca_metadata &coords, const teca_metadata &atts, bool provides_time)
{
    // get the name and values
    std::string t_variable;
    const_p_teca_variant_array t;
    if (coords.get("t_variable", t_variable) || !(t = coords.get("t")))
    {
        TECA_ERROR("Failed to get attributes for the time axis  \"" << t_variable
            << "\" from source \"" << source << ". A validation is not possible.")
        return -1;
    }

    // get calendaring info, but don't error out if it's not present
    teca_metadata t_atts;
    atts.get(t_variable, t_atts);

    std::string t_units;
    std::string calendar;
    t_atts.get("calendar", calendar);
    t_atts.get("units", t_units);

    // save it for the validation step
    if (provides_time)
    {
        m_t.set_reference_array(source, t_variable, t_units + " " + calendar, t);
    }
    else
    {
        m_t.append_array(source, t_variable, t_units + " " + calendar, t);
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_coordinate_axis_validator::add_x_coordinate_axis(const std::string &source,
    const teca_metadata &coords, const teca_metadata &atts, bool provides_geometry)
{
    // get the name and values
    std::string x_variable;
    const_p_teca_variant_array x;
    if (coords.get("x_variable", x_variable) || !(x = coords.get("x")))
    {
        TECA_ERROR("Failed to get attributes for the x-coordinate axis  \"" << x_variable
            << "\" from source \"" << source << ". A validation is not possible.")
        return -1;
    }

    // get the units, not an error if they are not present
    teca_metadata x_atts;
    std::string x_units;
    atts.get(x_variable, x_atts);
    x_atts.get("units", x_units);

    // save for later validation
    if (provides_geometry)
    {
        m_x.set_reference_array(source, x_variable, x_units, x);
    }
    else
    {
        m_x.append_array(source, x_variable, x_units, x);
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_coordinate_axis_validator::add_y_coordinate_axis(const std::string &source,
    const teca_metadata &coords, const teca_metadata &atts, bool provides_geometry)
{
    // get the name and values
    std::string y_variable;
    const_p_teca_variant_array y;
    if (coords.get("y_variable", y_variable) || !(y = coords.get("y")))
    {
        TECA_ERROR("Failed to get attributes for the y-coordinate axis  \"" << y_variable
            << "\" from source \"" << source << ". A validation is not possible.")
        return -1;
    }

    // get the units
    teca_metadata y_atts;
    std::string y_units;
    atts.get(y_variable, y_atts);
    y_atts.get("units", y_units);

    // save for later validation
    if (provides_geometry)
    {
        m_y.set_reference_array(source, y_variable, y_units, y);
    }
    else
    {
        m_y.append_array(source, y_variable, y_units, y);
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_coordinate_axis_validator::add_z_coordinate_axis(const std::string &source,
    const teca_metadata &coords, const teca_metadata &atts, bool provides_geometry)
{
    // get the name and values
    std::string z_variable;
    const_p_teca_variant_array z;
    if (coords.get("z_variable", z_variable) || !(z = coords.get("z")))
    {
        TECA_ERROR("Failed to get attributes for the z-coordinate axis  \"" << z_variable
            << "\" from source \"" << source << ". A validation is not possible.")
        return -1;
    }

    // get the untis
    teca_metadata z_atts;
    std::string z_units;

    atts.get(z_variable, z_atts);
    z_atts.get("units", z_units);

    // save for later validation
    if (provides_geometry)
    {
        m_z.set_reference_array(source, z_variable, z_units, z);
    }
    else
    {
        m_z.append_array(source, z_variable, z_units, z);
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_coordinate_axis_validator::validate_spatial_coordinate_axes(std::string &errorStr)
{
    int errorNo = 0;

    if ((errorNo = m_x.validate("x-coordinate axis",
        m_absolute_tolerance, m_relative_tolerance, errorStr)) ||
        (errorNo = m_y.validate("y-coordinate axis",
        m_absolute_tolerance, m_relative_tolerance, errorStr)) ||
        (errorNo = m_z.validate("z-coordinate axis",
        m_absolute_tolerance, m_relative_tolerance, errorStr)))
    {
        return errorNo;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_coordinate_axis_validator::validate_time_axis(std::string &errorStr)
{
    return m_t.validate("time axis",
        m_absolute_tolerance, m_relative_tolerance, errorStr);
}

};
