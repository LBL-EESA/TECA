#ifndef gfdl_tc_candidates_h
#define gfdl_tc_candidates_h

// declares the fortran function and a c++ overload to forward
// to it
#define DECLARE_GFDL_TC_CANDIDATES(_c_type, _c_name, _v_type, _v_name) \
\
extern "C" \
int gfdl_tc_candidates_c ## _c_name ## _v ## _v_name ( \
     const _v_type *core_rad, const _v_type *min_vort, \
     const _v_type *vort_win, const _v_type *max_psl_dy, \
     const _v_type *max_psl_dr, const _v_type *max_twc_dy, \
     const _v_type *max_twc_dr, const _v_type *max_thick_dy, \
     const _v_type *max_thick_dr, const _v_type *Gwind, \
     const _v_type *Gvort, const _v_type *Gtbar, \
     const _v_type *Gpsl, const _v_type *Gthick, \
     const _c_type *Grlat, const _c_type *Grlon, long *Gnlat, \
     long *Gnlon, int *frprm_itmax, long *step, void *atable); \
\
namespace teca_gfdl { \
int tc_candidates( \
     _v_type core_rad, _v_type min_vort, _v_type vort_win, \
     _v_type max_psl_dy, _v_type max_psl_dr, _v_type max_twc_dy, \
     _v_type max_twc_dr, _v_type max_thick_dy, _v_type max_thick_dr, \
     const _v_type *Gwind, const _v_type *Gvort, const _v_type *Gtbar, \
     const _v_type *Gpsl, const _v_type *Gthick, const _c_type *Grlat, \
     const _c_type *Grlon, long Gnlat, long Gnlon, int frprm_itmax, \
     long step, void *atable) \
{ \
    return gfdl_tc_candidates_c ## _c_name ## _v ## _v_name ( \
        &core_rad, &min_vort, &vort_win, &max_psl_dy, &max_psl_dr, \
        &max_twc_dy, &max_twc_dr, &max_thick_dy, &max_thick_dr, \
        Gwind, Gvort, Gtbar, Gpsl, Gthick, Grlat, Grlon, &Gnlat, \
        &Gnlon, &frprm_itmax, &step, atable); \
} \
};

DECLARE_GFDL_TC_CANDIDATES(float, f, float, f)
DECLARE_GFDL_TC_CANDIDATES(float, f, double, d)
DECLARE_GFDL_TC_CANDIDATES(double, d, float, f)
DECLARE_GFDL_TC_CANDIDATES(double, d, double, d)

#endif
