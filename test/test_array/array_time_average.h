#ifndef array_time_average_h
#define array_time_average_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(array_time_average)

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

private:
    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string array_name;
    unsigned int filter_width;
};

#endif
