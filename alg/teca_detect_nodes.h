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

    TECA_ALGORITHM_PROPERTY(std::string, in_connect)
    TECA_ALGORITHM_PROPERTY(std::string, searchbymin)
    TECA_ALGORITHM_PROPERTY(std::string, searchbymax)
    TECA_ALGORITHM_PROPERTY(std::string, closedcontourcmd)
    TECA_ALGORITHM_PROPERTY(std::string, noclosedcontourcmd)
    TECA_ALGORITHM_PROPERTY(std::string, thresholdcmd)
    TECA_ALGORITHM_PROPERTY(std::string, outputcmd)
    TECA_ALGORITHM_PROPERTY(std::string, searchbythreshold)
    TECA_ALGORITHM_PROPERTY(double, minlon)
    TECA_ALGORITHM_PROPERTY(double, maxlon)
    TECA_ALGORITHM_PROPERTY(double, minlat)
    TECA_ALGORITHM_PROPERTY(double, maxlat)
    TECA_ALGORITHM_PROPERTY(double, minabslat)
    TECA_ALGORITHM_PROPERTY(double, mergedist)
    TECA_ALGORITHM_PROPERTY(bool, diag_connect)
    TECA_ALGORITHM_PROPERTY(bool, regional)
    TECA_ALGORITHM_PROPERTY(bool, out_header)

protected:
    teca_detect_nodes();

    // helper that computes the output extent
    int get_active_extent(
        const const_p_teca_variant_array &lat,
        const const_p_teca_variant_array &lon,
        std::vector<unsigned long> &extent) const;

    int DetectCyclonesUnstructured(
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
    std::string searchbymin;
    std::string searchbymax;
    std::string closedcontourcmd;
    std::string noclosedcontourcmd;
    std::string thresholdcmd;
    std::string outputcmd;
    std::string searchbythreshold;
    double minlon;
    double maxlon;
    double minlat;
    double maxlat;
    double minabslat;
    double mergedist;
    bool diag_connect;
    bool regional;
    bool out_header;

    class internals_t;
    internals_t *internals;
};
#endif
