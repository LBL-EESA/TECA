#ifndef teca_temporal_average_h
#define teca_temporal_average_h

#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <memory>
#include <string>
#include <vector>

class teca_temporal_average;
using p_teca_temporal_average = std::shared_ptr<teca_temporal_average>;
using const_p_teca_temporal_average = std::shared_ptr<teca_temporal_average>;

/// an algorithm that averages data in time
/**
an algorithm that averages data in time. filter_width
controls the number of time steps to average over.
all arrays in the input data are processed.
*/
class teca_temporal_average : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_temporal_average)
    ~teca_temporal_average();

    // set the number of steps to average. should be odd.
    TECA_ALGORITHM_PROPERTY(unsigned int, filter_width)

protected:
    teca_temporal_average();

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
    unsigned int filter_width;
};

#endif
