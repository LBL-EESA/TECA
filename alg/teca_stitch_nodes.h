#ifndef teca_stitch_nodes_h
#define teca_stitch_nodes_h

#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_stitch_nodes)

/**
 * Porting the StitchNodes function from TempestExtremes to TECA.
 * stitch_nodes is used to connect nodal features together in time,
 * producing paths associated with singular features.
 * Additional filtering of the output of detect_nodes can be applied
 * based on the temporal features of these paths.
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

    /** @name in_connet
     * Set the connectivity file
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, in_connect)
    ///@}

    /** @name in_fmt
     * Tracks output commands
     * [var,op,dist;...]
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, in_fmt)
    ///@}

    /** @name min_time
     * Minimum duration of path
     * default 10
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, min_time)
    ///@}

    /** @name cal_type
     * Calendar type
     * default standard
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, cal_type)
    ///@}

    /** @name max_gap
     * Maximum time gap (in time steps or duration)
     * default 3
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, max_gap)
    ///@}

    /** @name track_threshold_cmd
     * Threshold commands for path
     *[var,op,value,count;...]
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, track_threshold_cmd)
    ///@}

    /** @name prioritize
     * Variable to use when prioritizing paths
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, prioritize)
    ///@}

    /** @name min_path_length
     * Minimum path length
     * default 0
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, min_path_length)
    ///@}

    /** @name range
     * Range (in degrees)
     * default 8
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, range)
    ///@}

    /** @name min_endpoint_distance
     * Minimum distance between endpoints of path
     * default 0
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, min_endpoint_distance)
    ///@}

    /** @name min_path_distance
     * Minimum path lengt
     * default 0
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(double, min_path_distance)
    ///@}

    /** @name allow_repeated_times
     * Allow repeated times
     * default false
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(bool, allow_repeated_times)
    ///@}

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
    std::string track_threshold_cmd;
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
