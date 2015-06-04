#ifndef teca_temporal_average_h
#define teca_temporal_average_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_temporal_average)

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

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the number of steps to average. should be odd.
    TECA_ALGORITHM_PROPERTY(unsigned int, filter_width)


    // select the filter stencil, default is backward
    enum {
        backward,
        centered,
        forward
    };
    TECA_ALGORITHM_PROPERTY(int, filter_type)

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
    int filter_type;
};

#endif
