#ifndef gfdl_trajectory_h
#define gfdl_trajectory_h

#include "teca_table.h"

#define DECLARE_GFDL_TC_TRAJECTORY(_c_type, _c_name, _v_type, _v_name) \
\
extern "C" \
int gfdl_tc_trajectory_c ## _c_name ## _v ## _v_name ( \
    _c_type *r_crit, _v_type *wind_crit, _v_type *n_wind_crit, \
    const long *step, const double *time, const int *storm_id, \
    const _c_type *rlon, const _c_type *rlat, const _v_type *wind_max, \
    const _v_type *vort_max, const _v_type *psl, const int *have_twc, \
    const int *have_thick, const _v_type *twc_max, \
    const _v_type *thick_max, long *n_rows, void *track_table); \
\
namespace teca_gfdl { \
int tc_trajectory(_c_type r_crit, _v_type wind_crit, _v_type n_wind_crit, \
    const long *step, const double *time, const int *storm_id, \
    const _c_type *rlon, const _c_type *rlat, const _v_type *wind_max, \
    const _v_type *vort_max, const _v_type *psl, const int *have_twc, \
    const int *have_thick, const _v_type *twc_max, \
    const _v_type *thick_max, long n_rows, p_teca_table &track_table) \
{ \
    return gfdl_tc_trajectory_c ## _c_name ## _v ## _v_name ( \
        &r_crit, &wind_crit, &n_wind_crit, step, time, storm_id, \
        rlon, rlat, wind_max, vort_max, psl, have_twc, have_thick, \
        twc_max, thick_max, &n_rows, track_table.get()); \
} \
};

DECLARE_GFDL_TC_TRAJECTORY(float, f, float, f)
DECLARE_GFDL_TC_TRAJECTORY(double, d, float, f)
DECLARE_GFDL_TC_TRAJECTORY(float, f, double, d)
DECLARE_GFDL_TC_TRAJECTORY(double, d, double, d)

#endif
