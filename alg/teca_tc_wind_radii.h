#ifndef teca_tc_wind_radii_h
#define teca_tc_wind_radii_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_tc_wind_radii)

/// computes wind radius at the specified coordinates
/**
Compute storm size and adds it to the table. There are two inputs,
the first serves up tables of storms to compute the storm radius
for. One must set the names of the columns that contain storm ids,
x-coordnates, y-coordinates, and time coordinate. For each event
the wind radius is computed. Computations are parallelized over
storm id. The second input serves up wind velocity data most likely
this will be from a NetCDF CF2 simulation dataset. By default
radius is computed at the transitions on the Saffir-Simpson
scale. 
*/
class teca_tc_wind_radii : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_tc_wind_radii)
    ~teca_tc_wind_radii();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the name of the column that defines the track ids
    // if set the specified column is coppied into the output
    // metadata and accessed with the key event_id
    TECA_ALGORITHM_PROPERTY(std::string, storm_id_column)

    // set the name of the columns that define the event position
    // if  set the columns are coppied into the output metadata
    // and accessed with the keys storm_x_coordinate, storm_y_coordinate
    TECA_ALGORITHM_PROPERTY(std::string, storm_x_coordinate_column)
    TECA_ALGORITHM_PROPERTY(std::string, storm_y_coordinate_column)

    // set the name of the column containing peak instantanious
    // surface wind speed
    TECA_ALGORITHM_PROPERTY(std::string, storm_wind_speed_column)

    // set the name of the column that defines the event time
    // if set the specified column is coppied into the output
    // metadata and accessed with the key event_time
    TECA_ALGORITHM_PROPERTY(std::string, storm_time_column)

    // set the name of the wind variable components
    TECA_ALGORITHM_PROPERTY(std::string, wind_u_variable)
    TECA_ALGORITHM_PROPERTY(std::string, wind_v_variable)

    // set the radius in degrees of latitude to sample the wind
    // field
    TECA_ALGORITHM_PROPERTY(double, search_radius)

    // set the radius in degrees of latitude beyond which to
    // terminate the search for peak wind speed. if the peak
    // lies beyond this distance search is terminated and a
    // warning is displayed.
    TECA_ALGORITHM_PROPERTY(double, core_radius)

    // number of bins to discetize by in the radial direction
    TECA_ALGORITHM_PROPERTY(int, number_of_radial_bins)

    // set the wind speeds (in m/s) to find the radius of. the
    // defualt values are the transition speeds of the Saffir-Simpson
    // scale.
    TECA_ALGORITHM_VECTOR_PROPERTY(double, critical_wind_speed)

    // set the profile type. PROFILE_MAX uses the maximum
    // wind speed on each interval of the discretization, while
    // PROFILE_AVERAGE uses the average on each interval
    enum {PROFILE_MAX = 0, PROFILE_AVERAGE = 1};
    TECA_ALGORITHM_PROPERTY(int, profile_type);


    // override the input connections because we are going to
    // take the first input and use it to generate metadata.
    // the second input then becomes the only one the pipeline
    // knows about.
    void set_input_connection(unsigned int id,
        const teca_algorithm_output_port &port) override;

protected:
    teca_tc_wind_radii();

private:
    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port, const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void set_modified() override;

private:
    // for the metadata input
    std::string storm_id_column;
    std::string storm_x_coordinate_column;
    std::string storm_y_coordinate_column;
    std::string storm_wind_speed_column;
    std::string storm_time_column;

    // for netcdf cf data input
    std::string wind_u_variable;
    std::string wind_v_variable;

    std::vector<double> critical_wind_speeds;
    double search_radius;
    double core_radius;
    int number_of_radial_bins;
    int profile_type;

    class internals_t;
    internals_t *internals;
};

#endif
