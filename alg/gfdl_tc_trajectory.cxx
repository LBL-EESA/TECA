#include "teca_table.h"

#define DECLARE_TECA_APPEND_TC_TRACK(_c_type, _c_name, _v_type, _v_name) \
\
void teca_append_tc_track_c ## _c_name ## _v ## _v_name ( \
    void *vp_table, int *track_id, int *storm_id, long *step, \
    double *time, _c_type *lon, _c_type *lat, _v_type *psl, \
    _v_type *wind_max, _v_type *vort_max, int *have_twc, \
    int *have_thick, _v_type *twc_max, _v_type *thick_max) \
{ \
    teca_table *p_table = static_cast<teca_table*>(vp_table); \
    p_table->append(*track_id, *storm_id, *step, *time, \
    *lon, *lat, *psl, *wind_max, *vort_max, *have_twc, \
    *have_thick, *twc_max, *thick_max); \
}

extern "C" {

DECLARE_TECA_APPEND_TC_TRACK(float, f, float, f)
DECLARE_TECA_APPEND_TC_TRACK(float, f, double, d)
DECLARE_TECA_APPEND_TC_TRACK(double, d, float, f)
DECLARE_TECA_APPEND_TC_TRACK(double, d, double, d)

};
