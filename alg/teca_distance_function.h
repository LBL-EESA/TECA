#ifndef teca_distance_function_h
#define teca_distance_function_h

#include <cmath>

// --------------------------------------------------------------------------
template<typename coord_t>
coord_t teca_distance(coord_t r_lon_0, coord_t r_lat_0,
    coord_t r_lon_i, coord_t r_lat_i)
{
    const coord_t EARTH_RADIUS = 6371.0;

    // GFDL 2007
    coord_t dx = (r_lon_i - r_lon_0)*std::cos(r_lat_0);
    coord_t dy = (r_lat_i - r_lat_0);
    return EARTH_RADIUS*std::sqrt(dx*dx + dy*dy);

    // GFDL 2015
    //dr = EARTH_RADIUS*acos(sin(r_lat_0)*sin(r_lat_i) &
    //  + cos(r_lat_0)*cos(r_lat_i)*cos(r_lon_i - r_lon_0))

    // haversine formula
    //dx = r_lon_i - r_lon_0
    //dy = r_lat_i - r_lat_0
    //dr = EARTH_RADIUS*2.0*asin(min(1,sqrt(sin(dy/2.0)**2.0 &
    //  + cos(r_lat_0)*cos(r_lat_i)*sin(dx/2.0)**2.0)))
}

#endif
