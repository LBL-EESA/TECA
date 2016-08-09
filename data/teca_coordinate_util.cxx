#include "teca_coordinate_util.h"

#include "teca_common.h"
#if defined(TECA_HAS_UDUNITS)
#include "calcalcs.h"
#endif

#include <string>
#include <cstdio>

namespace teca_coordinate_util
{

// **************************************************************************
int time_step_of(p_teca_double_array time, bool lower,
    const std::string &calendar, const std::string &units,
    const std::string &date, unsigned long &step)
{
#if defined(TECA_HAS_UDUNITS)
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

    double t = 0.0;
    if (calcalcs::coordinate(Y, M, D, h, m, s,
        units.c_str(), calendar.c_str(), &t))
    {
        TECA_ERROR("failed to convert date \"" << date
            << "\" to relative time in the \"" << calendar
            << "\" calendar in units of \"" << units << "\".")
        return -1;
    }

    step = 0;
    unsigned long last = time->size() - 1;
    if (teca_coordinate_util::index_of(time->get(), 0, last, t, lower, step))
    {
        TECA_ERROR("failed to locate the requested time " << t << " in ["
            << time->get(0) << ", " << time->get(last) << "]")
        return -1;
    }

    return 0;
#else
    (void)time;
    (void)lower;
    (void)calendar;
    (void)units;
    (void)date;
    step = 0;
    TECA_ERROR("The UDUnits package is required for this operation")
    return -1;
#endif
}

// **************************************************************************
int bounds_to_extent(const double *bounds,
    const_p_teca_variant_array x, const_p_teca_variant_array y,
    const_p_teca_variant_array z, unsigned long *extent)
{
    TEMPLATE_DISPATCH_FP(
        const teca_variant_array_impl,
        x.get(),

        unsigned long nx = x->size();
        unsigned long high_i = nx - 1;
        extent[0] = 0;
        extent[1] = high_i;
        const NT *p_x = std::dynamic_pointer_cast<TT>(x)->get();

        unsigned long ny = y->size();
        unsigned long high_j = ny - 1;
        extent[2] = 0;
        extent[3] = high_j;
        const NT *p_y = std::dynamic_pointer_cast<TT>(y)->get();

        unsigned long nz = z->size();
        unsigned long high_k = nz - 1;
        extent[4] = 0;
        extent[5] = high_k;
        const NT *p_z = std::dynamic_pointer_cast<TT>(z)->get();

        if (((nx > 1) && (teca_coordinate_util::index_of(p_x, 0, high_i, static_cast<NT>(bounds[0]), true, extent[0])
            || teca_coordinate_util::index_of(p_x, 0, high_i, static_cast<NT>(bounds[1]), false, extent[1]))) ||
            ((ny > 1) && (teca_coordinate_util::index_of(p_y, 0, high_j, static_cast<NT>(bounds[2]), true, extent[2])
            || teca_coordinate_util::index_of(p_y, 0, high_j, static_cast<NT>(bounds[3]), false, extent[3]))) ||
            ((nz > 1) && (teca_coordinate_util::index_of(p_z, 0, high_k, static_cast<NT>(bounds[4]), true, extent[4])
            || teca_coordinate_util::index_of(p_z, 0, high_k, static_cast<NT>(bounds[5]), false, extent[5]))))
        {
            TECA_ERROR(<< "requested subset [" << bounds[0] << ", " << bounds[1] << ", "
                << bounds[2] << ", " << bounds[3] << ", " << bounds[4] << ", " << bounds[5]
                << "] is not contained in the current dataset bounds ["
                << p_x[0] << ", " << p_x[high_i] << ", " << p_y[0] << ", " << p_y[high_j] << ", "
                << p_z[0] << ", " << p_y[high_k] << "]")
            return -1;
        }
        return 0;
        )

    TECA_ERROR("invalid coordinate array type")
    return -1;
}
};
