#ifndef array_time_average_h
#define array_time_average_h

#include "teca_algorithm.h"
#include "teca_meta_data.h"

#include <memory>
#include <string>
#include <vector>

class array_time_average;
typedef std::shared_ptr<array_time_average> p_array_time_average;

/**
an example implementation of a teca_algorithm
that avergaes n timesteps
*/
class array_time_average : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(array_time_average)
    ~array_time_average();

    // set the name of the array to average
    TECA_ALGORITHM_PROPERTY(std::string, array_name)

    // set the number of steps to average. should be odd.
    TECA_ALGORITHM_PROPERTY(unsigned int, filter_width)

protected:
    array_time_average();

    // helper that creates a list of times required
    // to compute the average
    int get_active_times(
        const teca_meta_data &request,
        const teca_meta_data &input_md,
        double &current_time,
        std::vector<double> &active_times);

private:
    /*virtual
    teca_meta_data get_output_meta_data(
        unsigned int port,
        std::vector<teca_meta_data> &input_md);*/

    virtual
    std::vector<teca_meta_data> get_upstream_request(
        unsigned int port,
        std::vector<teca_meta_data> &input_md,
        teca_meta_data &request);

    virtual
    p_teca_dataset execute(
        unsigned int port,
        std::vector<p_teca_dataset> &input_data,
        teca_meta_data &request);

private:
    std::string array_name;
    unsigned int filter_width;
};

#endif
