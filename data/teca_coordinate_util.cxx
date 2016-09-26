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
        const NT *px = std::dynamic_pointer_cast<TT>(x)->get();
        NT low_x = static_cast<NT>(bounds[0]);
        NT high_x = static_cast<NT>(bounds[1]);
        bool slice_x = equal(low_x, high_x, eps8);

        unsigned long ny = y->size();
        unsigned long high_j = ny - 1;
        extent[2] = 0;
        extent[3] = high_j;
        const NT *py = std::dynamic_pointer_cast<TT>(y)->get();
        NT low_y = static_cast<NT>(bounds[2]);
        NT high_y = static_cast<NT>(bounds[3]);
        bool slice_y = equal(low_y, high_y, eps8);

        unsigned long nz = z->size();
        unsigned long high_k = nz - 1;
        extent[4] = 0;
        extent[5] = high_k;
        const NT *pz = std::dynamic_pointer_cast<TT>(z)->get();
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

    TECA_ERROR("invalid coordinate array type")
    return -1;
}
};
