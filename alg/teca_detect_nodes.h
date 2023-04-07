#ifndef teca_detect_nodes_h
#define teca_detect_nodes_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_table.h"
#include "teca_coordinate_util.h"

#include <SimpleGrid.h>

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_detect_nodes)

/**
 *
 *
 *
 */
class TECA_EXPORT teca_detect_nodes : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_detect_nodes)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_detect_nodes)
    TECA_ALGORITHM_CLASS_NAME(teca_detect_nodes)
    ~teca_detect_nodes();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    int initialize();

    TECA_ALGORITHM_PROPERTY(std::string, in_connect)
    TECA_ALGORITHM_PROPERTY(std::string, search_by_min)
    TECA_ALGORITHM_PROPERTY(std::string, search_by_max)
    TECA_ALGORITHM_PROPERTY(std::string, closed_contour_cmd)
    TECA_ALGORITHM_PROPERTY(std::string, no_closed_contour_cmd)
    TECA_ALGORITHM_PROPERTY(std::string, threshold_cmd)
    TECA_ALGORITHM_PROPERTY(std::string, output_cmd)
    TECA_ALGORITHM_PROPERTY(std::string, search_by_threshold)
    TECA_ALGORITHM_PROPERTY(double, min_lon)
    TECA_ALGORITHM_PROPERTY(double, max_lon)
    TECA_ALGORITHM_PROPERTY(double, min_lat)
    TECA_ALGORITHM_PROPERTY(double, max_lat)
    TECA_ALGORITHM_PROPERTY(double, min_abs_lat)
    TECA_ALGORITHM_PROPERTY(double, merge_dist)
    TECA_ALGORITHM_PROPERTY(bool, diag_connect)
    TECA_ALGORITHM_PROPERTY(bool, regional)
    TECA_ALGORITHM_PROPERTY(bool, out_header)

protected:
    teca_detect_nodes();

    int detect_cyclones_unstructured(
        const_p_teca_cartesian_mesh mesh,
        SimpleGrid & grid,
        std::set<int> & setCandidates);

    std::vector<teca_metadata> get_upstream_request(
        unsigned int port,
        const std::vector<teca_metadata> &md_in,
        const teca_metadata &req_in) override;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &req_in) override;

private:
    std::string in_connect;
    std::string search_by_min;
    std::string search_by_max;
    std::string closed_contour_cmd;
    std::string no_closed_contour_cmd;
    std::string threshold_cmd;
    std::string output_cmd;
    std::string search_by_threshold;
    double min_lon;
    double max_lon;
    double min_lat;
    double max_lat;
    double min_abs_lat;
    double merge_dist;
    bool diag_connect;
    bool regional;
    bool out_header;

    class internals_t;
    internals_t *internals;
};
#endif
