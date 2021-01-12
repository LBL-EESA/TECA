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
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_tc_wind_radii)
    TECA_ALGORITHM_CLASS_NAME(teca_tc_wind_radii)
    ~teca_tc_wind_radii();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    /** @anchor storm_id_column
     * @name storm_id_column
     * Set the name of the column that defines the track ids
     * if set the specified column is coppied into the output
     * metadata and accessed with the key event_id.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, storm_id_column)
    ///@}

    /** @anchor storm_x_coordinate_column
     * @name storm_x_coordinate_column
     * Set the name of the `storm_x_coordinate_column`
     * column that define the event position.
     * If set the column is copied into the output metadata
     * and accessed with the key storm_x_coordinate.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, storm_x_coordinate_column)
    ///@}

    /** @anchor storm_y_coordinate_column
     * @name storm_y_coordinate_column
     * Set the name of the `storm_y_coordinate_column`
     * column that define the event position.
     * If set the column is copied into the output metadata
     * and accessed with the key storm_y_coordinate.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, storm_y_coordinate_column)
    ///@}

    /** @anchor storm_wind_speed_column
     * @name storm_wind_speed_column
     * Set the name of the column containing peak instantanious
     * surface wind speed
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, storm_wind_speed_column)
    ///@}

    /** @anchor storm_time_column
     * @name storm_time_column
     * Set the name of the column that defines the event time
     * if set the specified column is coppied into the output
     * metadata and accessed with the key event_time
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, storm_time_column)
    ///@}

    /** @anchor wind_u_variable
     * @name wind_u_variable
     * Set the name of the U wind variable component.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, wind_u_variable)
    ///@}

    /** @anchor wind_v_variable
     * @name wind_v_variable
     * Set the name of the V wind variable component.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, wind_v_variable)
    ///@}

    /** @anchor search_radius
     * @name search_radius
     * Set the radius in degrees of latitude to sample the wind field.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, search_radius)
    ///@}

    /** @anchor core_radius
     * @name core_radius
     * Set the radius in degrees of latitude beyond which to
     * terminate the search for peak wind speed. if the peak
     * lies beyond this distance search is terminated and a
     * warning is displayed.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, core_radius)
    ///@}

    /** @anchor number_of_radial_bins
     * @name number_of_radial_bins
     * Set the number of bins to discetize by in the radial direction.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, number_of_radial_bins)
    ///@}

    /** @anchor critical_wind_speeds
     * @name critical_wind_speeds
     * Set the wind speeds (in m/s) to find the radius of. the
     * defualt values are the transition speeds of the Saffir-Simpson
     * scale.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(double, critical_wind_speed)
    ///@}

    /** Profile types. PROFILE_MAX uses the maximum
     * wind speed on each interval of the discretization, while
     * PROFILE_AVERAGE uses the average on each interval
     */
    enum {
        /** 0 */
        PROFILE_MAX = 0,
        /** 1 */
        PROFILE_AVERAGE = 1
    };

    /** @anchor profile_type
     * @name profile_type
     * Set the profile type.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, profile_type);
    ///@}


    /** override the input connections because we are going to
     * take the first input and use it to generate metadata.
     * the second input then becomes the only one the pipeline
     * knows about.
     */
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
