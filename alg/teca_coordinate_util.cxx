#include "teca_coordinate_util.h"

#include "teca_common.h"
#include "calcalcs.h"

#include <string>
#include <cstdio>

namespace teca_coordinate_util
{
int time_step_of(p_teca_double_array time, bool lower,
    const std::string &calendar, const std::string &units,
    const std::string &date, unsigned long &step)
{
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
}
};
