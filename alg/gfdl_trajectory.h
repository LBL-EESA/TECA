#ifndef gfdl_trajectory_h
#define gfdl_trajectory_h
extern "C" {
// --------------------------------------------------------------------------
int gfdl_tc_trajectory_float(const void *tc_summary, const void *tc_details,
    float *rcrit, float *wcrit, float *wcritm, float *nwcrit, float *vcrit,
    float *twc_crit, float *thick_crit, float *nlat, float *slat,
    int *do_spline, int *do_thickness, void *traj_summary, void *traj_details);
};

namespace {
// --------------------------------------------------------------------------
template<typename num_t>
int get_column(teca_table *table, long row_i, long n_rows,
    const char *col_id, num_t *array)
{
    p_teca_variant_array col = table->get_column(col_id);
    if (!col)
    {
        TECA_ERROR("Failed to locate column " << col_id)
        return -1;
    }
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        col.get(),
        TT *colt = static_cast<TT*>(col.get());
        colt->get(row_i, row_i + n_rows - 1, array);
        return 0;
        )
    TECA_ERROR("Unsupported type")
    return -1;
}

// --------------------------------------------------------------------------
template<typename num_t>
void append_trajectory_details(teca_table *table, num_t *lon, num_t *lat,
    num_t *wind, num_t *psl, num_t *vmax, int *year, int *month, int *day,
    int *hour)
{
    table->append(*lon, *lat, *wind, *psl,
         *vmax, *year, *month, *day, *hour);
}

// --------------------------------------------------------------------------
template<typename num_t>
void append_trajectory_summary(teca_table *table, int *num_steps,
    int *year, int *month, int *day, int *hour, num_t *lon, num_t *lat)
{
    table->append(*num_steps, *year, *month, *day, *hour, *lon, *lat);
}


// --------------------------------------------------------------------------
int gfdl_tc_trajectory(const_p_teca_table tc_summary,
    const_p_teca_table tc_details,
    float rcrit, float wcrit, float wcritm, float nwcrit,
    float vcrit, float twc_crit, float thick_crit, float nlat,
    float slat, int do_spline, int do_thickness,
    p_teca_table traj_summary, p_teca_table traj_details)
{
    return gfdl_tc_trajectory_float(
        tc_summary.get(), tc_details.get(), &rcrit, &wcrit, &wcritm,
        &nwcrit, &vcrit, &twc_crit, &thick_crit, &nlat, &slat, &do_spline,
        &do_thickness, traj_summary.get(), traj_details.get());
}
};

extern "C" {
// --------------------------------------------------------------------------
int teca_get_column_int(teca_table *table, long *row_i, long *n_rows,
    const char *col_id, int *array)
{
    return ::get_column(table, *row_i, *n_rows, col_id, array);
}

// --------------------------------------------------------------------------
int teca_get_column_long(teca_table *table, long *row_i, long *n_rows,
    const char *col_id, long *array)
{
    return ::get_column(table, *row_i, *n_rows, col_id, array);
}

// --------------------------------------------------------------------------
int teca_get_column_float(teca_table *table, long *row_i, long *n_rows,
    const char *col_id, float *array)
{
    return ::get_column(table, *row_i, *n_rows, col_id, array);
}

// --------------------------------------------------------------------------
int teca_get_column_double(teca_table *table, long *row_i, long *n_rows,
    const char *col_id, double *array)
{
    return ::get_column(table, *row_i, *n_rows, col_id, array);
}

// --------------------------------------------------------------------------
void teca_append_trajectory_summary_float(teca_table *table, int *num_steps,
    int *year, int *month, int *day, int *hour, float *start_lon,
    float *start_lat)
{
    ::append_trajectory_summary(table, num_steps,
        year, month, day, hour, start_lon, start_lat);
}

// --------------------------------------------------------------------------
void teca_append_trajectory_details_float(teca_table *table, float *lon,
    float *lat, float *wind, float *psl, float *vmax, int *year, int *month,
    int *day, int *hour)
{
    ::append_trajectory_details(table, lon, lat,
        wind, psl, vmax, year, month, day, hour);
}

// --------------------------------------------------------------------------
long teca_get_number_of_rows(teca_table *table)
{
    return table->get_number_of_rows();
}

};

#endif
