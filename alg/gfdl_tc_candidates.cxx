#include "teca_table.h"
#include "teca_common.h"

// this is the "fortran calls c++" part of the interface
// to the gfdl tstorms code

#define DECLARE_APPEND_CANDIDATE(_c_type, _c_name, _v_type, _v_name) \
\
void teca_tc_append_candidate_c ## _c_name ## _v ## _v_name ( \
    int *number, _c_type *spsl_lon, _c_type *spsl_lat, \
    _v_type *swind_max, _v_type *svort_max, _v_type *spsl_min, \
    int *stwc_is, int *sthick_is, _v_type *stwc_max, \
    _v_type *sthick_max, void *atable) \
{ \
    teca_table *table = static_cast<teca_table*>(atable); \
    table->append(*number, *spsl_lon, *spsl_lat, *swind_max, \
    *svort_max, *spsl_min, *stwc_is, *sthick_is, *stwc_max, \
    *sthick_max); \
}

extern "C" {

DECLARE_APPEND_CANDIDATE(float, f, float, f)
DECLARE_APPEND_CANDIDATE(float, f, double, d)
DECLARE_APPEND_CANDIDATE(double, d, float, f)
DECLARE_APPEND_CANDIDATE(double, d, double, d)

void teca_tc_status(const char *msg)
{ TECA_STATUS(<< msg) }

void teca_tc_warning(const char *msg)
{ TECA_WARNING(<< msg) }

void teca_tc_error(const char *msg)
{ TECA_ERROR(<< msg) }

};
