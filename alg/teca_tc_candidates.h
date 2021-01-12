#ifndef teca_tc_candidates_h
#define teca_tc_candidates_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_tc_candidates)

/**
GFDL tropical storms detection algorithm
for more information see
"Seasonal forecasting of tropical storms using coupled GCM integrations"

 --- INPUT
     Gwind  - wind speed at 850 mb
     Gvort  - vorticity_850mb  at 850 mb
     Gtbar  - mean core_temperature for warm core layer
     Gpsl   - sea level sea_level_pressure
     Gthick - thickness of 200 to 1000 mb layer
     Grlon  - longitudes
     Grlat  - latitudes
     iyear  - year
     imon   - month
     iday   - day of month
     ihour  - hour
     iucy   - unit for output

 --- OUTPUT
 --- record # 1
     num0   - day
     imon0  - month
     iyear  - year
     number - number of cyclones found
 --- records # 2...number+1
     idex, jdex - (i,j) index of cyclone
     svort_max  - max vorticity_850mb
     swind_max  - max wind
      spsl_min  - min sea level sea_level_pressure
     svort_lon,  svort_lat - longitude & latitude of max vorticity_850mb
      spsl_lon,   spsl_lat - longitude & latitude of min slp
      stemperature_lon,   stemperature_lat - longitude & latitude of warm core
    sthick_lon, sthick_lat - longitude & latitude of max thickness
*/
class teca_tc_candidates : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_tc_candidates)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_tc_candidates)
    TECA_ALGORITHM_CLASS_NAME(teca_tc_candidates)
    ~teca_tc_candidates();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @anchor surface_wind_speed_variable
     * @name surface_wind_speed_variable
     * Set the name of surface wind speed input variable
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, surface_wind_speed_variable)
    ///@}

    /** @anchor vorticity_850mb_variable
     * @name vorticity_850mb_variable
     * Set the name of vorticity 850mb input variable
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, vorticity_850mb_variable)
    ///@}

    /** @anchor sea_level_pressure_variable
     * @name sea_level_pressure_variable
     * Set the name of sea level pressure input variable
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, sea_level_pressure_variable)
    ///@}

    /** @anchor core_temperature_variable
     * @name core_temperature_variable
     * Set the name of core temperature input variable
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, core_temperature_variable)
    ///@}

    /** @anchor thickness_variable
     * @name thickness_variable
     * Set the name of thickness input variable
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, thickness_variable)
    ///@}

    /** @anchor max_core_radius
     * @name max_core_radius
     * Set `max_core_radius`.
     *
     * @note A candidate is defined as having:\n
     *       1) a local maximum in vorticity above vorticty_850mb_threshold,
     *          centered on a window of vorticty_850mb_window degrees\n
     *       2) a local minimum in pressure within max_core_radius degrees\n
     *       3) having max pressure delta within max_pressure_radius at
     *          that location
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, max_core_radius)
    ///@}

    /** @anchor min_vorticity_850mb
     * @name min_vorticity_850mb
     * Set `min_vorticity_850mb`.
     *
     * @note A candidate is defined as having:\n
     *       1) a local maximum in vorticity above vorticty_850mb_threshold,
     *          centered on a window of vorticty_850mb_window degrees\n
     *       2) a local minimum in pressure within max_core_radius degrees\n
     *       3) having max pressure delta within max_pressure_radius at
     *          that location
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, min_vorticity_850mb)
    ///@}

    /** @anchor vorticity_850mb_window
     * @name vorticity_850mb_window
     * Set `vorticity_850mb_window`.
     *
     * @note A candidate is defined as having:\n
     *       1) a local maximum in vorticity above vorticty_850mb_threshold,
     *          centered on a window of vorticty_850mb_window degrees\n
     *       2) a local minimum in pressure within max_core_radius degrees\n
     *       3) having max pressure delta within max_pressure_radius at
     *          that location
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, vorticity_850mb_window)
    ///@}

    /** @anchor max_pressure_delta
     * @name max_pressure_delta
     * Set `max_pressure_delta`.
     *
     * @note A candidate is defined as having:\n
     *       1) a local maximum in vorticity above vorticty_850mb_threshold,
     *          centered on a window of vorticty_850mb_window degrees\n
     *       2) a local minimum in pressure within max_core_radius degrees\n
     *       3) having max pressure delta within max_pressure_radius at
     *          that location
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, max_pressure_delta)
    ///@}

    /** @anchor max_pressure_radius
     * @name max_pressure_radius
     * Set `max_pressure_radius`.
     *
     * @note A candidate is defined as having:\n
     *       1) a local maximum in vorticity above vorticty_850mb_threshold,
     *          centered on a window of vorticty_850mb_window degrees\n
     *       2) a local minimum in pressure within max_core_radius degrees\n
     *       3) having max pressure delta within max_pressure_radius at
     *          that location
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, max_pressure_radius)
    ///@}

    /** @anchor max_core_temperature_delta
     * @name max_core_temperature_delta
     * this criteria is recorded here but only used in the trajectory
     * stitching stage.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, max_core_temperature_delta)
    ///@}

    /** @anchor max_core_temperature_radius
     * @name max_core_temperature_radius
     * this criteria is recorded here but only used in the trajectory
     * stitching stage.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, max_core_temperature_radius)
    ///@}

    /** @anchor max_thickness_delta
     * @name max_thickness_delta
     * this criteria is recorded here but only used in the trajectory
     * stitching stage.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, max_thickness_delta)
    ///@}

    /** @anchor max_thickness_radius
     * @name max_thickness_radius
     * this criteria is recorded here but only used in the trajectory
     * stitching stage.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, max_thickness_radius)
    ///@}

    /** @anchor search_lat_low
     * @name search_lat_low
     * Set the low lat bounding box to search for storms
     * in units of degrees
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, search_lat_low)
    ///@}

    /** @anchor search_lat_high
     * @name search_lat_high
     * Set the high lat bounding box to search for storms
     * in units of degrees
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, search_lat_high)
    ///@}

    /** @anchor search_lon_low
     * @name search_lon_low
     * Set the low lon bounding box to search for storms
     * in units of degrees
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, search_lon_low)
    ///@}

    /** @anchor search_lon_high
     * @name search_lon_high
     * Set the high lon bounding box to search for storms
     * in units of degrees
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, search_lon_high)
    ///@}

    /** @anchor minimizer_iterations
     * @name minimizer_iterations
     * Set the number of iterations to search for the
     * storm local minimum. raising this paramter might increase
     * detections but the detector will run slowerd. default is
     * 50.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, minimizer_iterations)
    ///@}

    /** send humand readable representation to the
     * stream
     */
    virtual void to_stream(std::ostream &os) const override;

protected:
    teca_tc_candidates();

    /** helper that computes the output extent */
    int get_active_extent(
        p_teca_variant_array lat,
        p_teca_variant_array lon,
        std::vector<unsigned long> &extent) const;

private:
    virtual
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    virtual
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    virtual
    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string surface_wind_speed_variable;
    std::string vorticity_850mb_variable;
    std::string sea_level_pressure_variable;
    std::string core_temperature_variable;
    std::string thickness_variable;

    double max_core_radius;
    double min_vorticity_850mb;
    double vorticity_850mb_window;
    double max_pressure_delta;
    double max_pressure_radius;
    double max_core_temperature_delta;
    double max_core_temperature_radius;
    double max_thickness_delta;
    double max_thickness_radius;

    double search_lat_low;
    double search_lat_high;
    double search_lon_low;
    double search_lon_high;

    int minimizer_iterations;
};

#endif
