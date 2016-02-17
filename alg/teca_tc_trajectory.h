#ifndef teca_tc_trajectory_h
#define teca_tc_trajectory_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_tc_trajectory)

/// GFDL tropical storms trajectory tracking algorithm
/**
GFDL tropical storms trajectory tracking algorithm

for more information see
"Seasonal forecasting of tropical storms using coupled GCM integrations"

computes trajectories from table of detections. trajectories
are stored in a table.

the trajectory computation makes use of the following paramteters:

max_daily_distance (900 km)
    event must be within this distance on the
    following day to be considered as part of the trajectory.
    rcrit

min_wind_speed (17 m/s)
    850 mb wind sped must be above this value.
    wcrit

min_peak_wind_speed (17 m/s)
    wind must exceed this value at least once in the storm
    wcritm

min_duration (2 days)
    criteria must be satisfied for this many days to be
    a candidate
    nwcrit

min_850mb_voriticity (3.5e-5)
    minimum 850 mb vorticty
    vcrit

core_temperature_delta (0.5 deg C)
    twc_crit

min_thickness (50 m)
    thick_crit

low_search_alttitude (-40 deg)
    slat

high_search_latitude (40 deg)
    nlat

use_splines (0)
    use spline fitting
    do_spline

use_thickness (0)
    use thickness criteria
    do_thickness
*/
class teca_tc_trajectory : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_tc_trajectory)
    ~teca_tc_trajectory();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the arrays that contain the vector components
    // to compute norm from
    TECA_ALGORITHM_PROPERTY(double, max_daily_distance)
    TECA_ALGORITHM_PROPERTY(double, min_wind_speed)
    TECA_ALGORITHM_PROPERTY(double, min_peak_wind_speed)
    TECA_ALGORITHM_PROPERTY(double, min_vorticity)
    TECA_ALGORITHM_PROPERTY(double, core_temperature_delta)
    TECA_ALGORITHM_PROPERTY(double, min_thickness)
    TECA_ALGORITHM_PROPERTY(double, min_duration)
    TECA_ALGORITHM_PROPERTY(double, low_search_latitude)
    TECA_ALGORITHM_PROPERTY(double, high_search_latitude)
    TECA_ALGORITHM_PROPERTY(int, use_splines)
    TECA_ALGORITHM_PROPERTY(int, use_thickness)

protected:
    teca_tc_trajectory();

private:
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    double max_daily_distance;
    double min_wind_speed;
    double min_peak_wind_speed;
    double min_vorticity;
    double core_temperature_delta;
    double min_thickness;
    double min_duration;
    double low_search_latitude;
    double high_search_latitude;
    int use_splines;
    int use_thickness;
};

#endif
