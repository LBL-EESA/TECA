#ifndef gfdl_tc_h
#define gfdl_tc_h

extern "C" {
// GFDL tstorms detector
int gfdl_tc_detect_f(
    float *crit_vort, float *crit_temp, float *crit_thick,
    float *crit_dist, int *do_spline, int *do_thickness,
    float *Gwind, float *Gvort, float *Gtbar, float *Gpsl,
    float *Gthick, float *Grlat, float *Grlon, long *Gnlat,
    long *Gnlon, int *count, void *atable);
};

namespace {
// append events to storm details table
template <typename num_t>
void teca_tc_append_details(int *number, int *idex, int *jdex,
    num_t *spsl_lon, num_t *spsl_lat, num_t *swind_max,
    num_t *svort_max, num_t *spsl_min, int *stwc_is,
    int *sthick_is, num_t *stwc_max, num_t *sthick_max,
    void *atable)
{
#if TECA_DEBUG > 1
    std::cerr << "apending event " << *number << std::endl;
    std::cerr << *idex << ", " <<  *jdex << ", " << *spsl_lon << ", "
        << *spsl_lat << ", " << *swind_max << ", " << *svort_max << ", "
        << *spsl_min << ", " << *stwc_is << ", " << *sthick_is << ", "
        << *stwc_max << ", " << *sthick_max << std::endl;
#else
    (void)number;
#endif
    teca_table *table = static_cast<teca_table*>(atable);
    table->append(*idex, *jdex, *spsl_lon, *spsl_lat,
        *swind_max, *svort_max, *spsl_min, *stwc_is,
        *sthick_is, *stwc_max, *sthick_max);
}

// wrapper
int tc_detect(float crit_vort, float crit_temp, float crit_thick,
    float crit_dist, int do_spline, int do_thickness,
    float *Gwind, float *Gvort, float *Gtbar, float *Gpsl,
    float *Gthick, float *Grlat, float *Grlon, long Gnlat,
    long Gnlon, int &count, void *atable)
{
    return gfdl_tc_detect_f(&crit_vort, &crit_temp, &crit_thick,
        &crit_dist, &do_spline, &do_thickness, Gwind, Gvort, Gtbar,
        Gpsl, Gthick, Grlat, Grlon, &Gnlat, &Gnlon, &count, atable);
}
};

extern "C" {
// append to storm details to the given table
void teca_tc_append_details_f(int *number, int *idex, int *jdex,
    float *spsl_lon, float *spsl_lat, float *swind_max,
    float *svort_max, float *spsl_min, int *stwc_is,
    int *sthick_is, float *stwc_max, float *sthick_max,
    void *atable)
{
    ::teca_tc_append_details(number, idex, jdex, spsl_lon, spsl_lat,
        swind_max, svort_max, spsl_min, stwc_is, sthick_is, stwc_max,
        sthick_max, atable);
}
};

#endif
