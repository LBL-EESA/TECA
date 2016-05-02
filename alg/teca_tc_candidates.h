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
    ~teca_tc_candidates();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set/get the name of input variables
    TECA_ALGORITHM_PROPERTY(std::string, surface_wind_speed_variable)
    TECA_ALGORITHM_PROPERTY(std::string, vorticity_850mb_variable)
    TECA_ALGORITHM_PROPERTY(std::string, sea_level_pressure_variable)
    TECA_ALGORITHM_PROPERTY(std::string, core_temperature_variable)
    TECA_ALGORITHM_PROPERTY(std::string, thickness_variable)

    // a candidate is defined as having:
    // 1) a local maximum in vorticity above vorticty_850mb_threshold,
    //    centered on a window of vorticty_850mb_window degrees
    // 2) a local minimum in pressure within max_core_radius degrees
    // 3) having max pressure delta within max_pressure_radius at
    //    that location
    TECA_ALGORITHM_PROPERTY(double, max_core_radius)
    TECA_ALGORITHM_PROPERTY(double, min_vorticity_850mb)
    TECA_ALGORITHM_PROPERTY(double, vorticity_850mb_window)
    TECA_ALGORITHM_PROPERTY(double, max_pressure_delta)
    TECA_ALGORITHM_PROPERTY(double, max_pressure_radius)

    // these criteria are recorded here but only used in the trajectory
    // stitching stage.
    TECA_ALGORITHM_PROPERTY(double, max_core_temperature_delta)
    TECA_ALGORITHM_PROPERTY(double, max_core_temperature_radius)
    TECA_ALGORITHM_PROPERTY(double, max_thickness_delta)
    TECA_ALGORITHM_PROPERTY(double, max_thickness_radius)

    // set/get the bounding box to search for storms
    // in units of degreees lat,lon
    TECA_ALGORITHM_PROPERTY(double, search_lat_low)
    TECA_ALGORITHM_PROPERTY(double, search_lat_high)
    TECA_ALGORITHM_PROPERTY(double, search_lon_low)
    TECA_ALGORITHM_PROPERTY(double, search_lon_high)

    // set/get the number of iterations to search for the
    // storm local minimum. raising this paramter might increase
    // detections but the detector will run slowerd. default is
    // 50.
    TECA_ALGORITHM_PROPERTY(int, minimizer_iterations)

    // send humand readable representation to the
    // stream
    virtual void to_stream(std::ostream &os) const override;

protected:
    teca_tc_candidates();

    // helper that computes the output extent
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
