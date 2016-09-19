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

max_daily_distance (1600 km)
    event must be within this distance on the
    following day to be considered as part of the trajectory.

min_wind_speed (17 m/s)
    850 mb wind sped must be above this value.

min_wind_duration (2 days)
    criteria must be satisfied for this many days to be
    a candidate
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
    TECA_ALGORITHM_PROPERTY(double, min_wind_duration)

    // number of time steps between candidate data
    // this is used to detect missing candidate data
    // and abort those tracks. default 1
    TECA_ALGORITHM_PROPERTY(unsigned long, step_interval)

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
    double min_wind_duration;
    unsigned long step_interval;
};

#endif
