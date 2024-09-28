#ifndef teca_stitch_nodes_h
#define teca_stitch_nodes_h

#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_stitch_nodes)

/**
 *
 *
 *
 */
class TECA_EXPORT teca_stitch_nodes : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_stitch_nodes)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_stitch_nodes)
    TECA_ALGORITHM_CLASS_NAME(teca_stitch_nodes)
    ~teca_stitch_nodes();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    int initialize();

    TECA_ALGORITHM_PROPERTY(std::string, in_connect)
    TECA_ALGORITHM_PROPERTY(std::string, in_fmt)
    TECA_ALGORITHM_PROPERTY(std::string, min_time)
    TECA_ALGORITHM_PROPERTY(std::string, cal_type)
    TECA_ALGORITHM_PROPERTY(std::string, max_gap)
    TECA_ALGORITHM_PROPERTY(std::string, threshold)
    TECA_ALGORITHM_PROPERTY(std::string, prioritize)
    TECA_ALGORITHM_PROPERTY(int, min_path_length)
    TECA_ALGORITHM_PROPERTY(double, range)
    TECA_ALGORITHM_PROPERTY(double, min_endpoint_distance)
    TECA_ALGORITHM_PROPERTY(double, min_path_distance)
    TECA_ALGORITHM_PROPERTY(bool, allow_repeated_times)

protected:
    teca_stitch_nodes();

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
    std::string in_connect;
    std::string in_fmt;
    std::string min_time;
    std::string cal_type;
    std::string max_gap;
    std::string threshold;
    std::string prioritize;
    int min_path_length;
    double range;
    double min_endpoint_distance;
    double min_path_distance;
    bool allow_repeated_times;

    class internals_t;
    internals_t *internals;
};
#endif
