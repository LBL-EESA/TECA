#ifndef teca_cf_space_time_time_step_mapper_h
#define teca_cf_space_time_time_step_mapper_h

#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_cf_layout_manager.h"
#include "teca_cf_time_step_mapper.h"
#include "teca_calendar_util.h"
#include "teca_mpi.h"
#include "teca_coordinate_util.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

class teca_cf_space_time_time_step_mapper;
using p_teca_cf_space_time_time_step_mapper = std::shared_ptr<teca_cf_space_time_time_step_mapper>;

/// NetCDF CF2 files time step mapper.
class TECA_EXPORT teca_cf_space_time_time_step_mapper : public teca_cf_time_step_mapper
{
public:

    /// allocate and return a new object
    static p_teca_cf_space_time_time_step_mapper New()
    { return p_teca_cf_space_time_time_step_mapper(new teca_cf_space_time_time_step_mapper); }

    ~teca_cf_space_time_time_step_mapper() {}

    /** initialize based on input metadata. this is a collective call creates
     * communicator groups for each file and creates the file layout managers
     * for the local rank. After this call one can access file managers to
     * create, define and write local datasets to the NetCDF files in cf
     * format.
     *
     * @param[in] comm the MPI communicator to parallelize execution across
     * @param[in] first_step the first step to process
     * @param[in] last_step the last step to process
     * @param[in] number_of_temporal_partitons the number of temporal
     *                                         partitions or zero to use the
     *                                         temporal_partiton_size parameter
     * @param[in] temporal_partition_size number of time steps per partition or
     *                                    zero to use the number_of_temporal
     *                                    partitons parameter.
     * @param[in] extent the spatial extent to partition [i0, i1, j0, j1, k0, k1]
     * @param[in] number_of_spatial_partitons the number of spatial partitons
     * @param[in] partition_x if zero skip splitting in the x-direction
     * @param[in] partition_y if zero skip splitting in the y-direction
     * @param[in] partition_z if zero skip splitting in the z-direction
     * @param[in] min_block_size_x sets the minimum block size in the x-direction
     * @param[in] min_block_size_y sets the minimum block size in the y-direction
     * @param[in] min_block_size_z sets the minimum block size in the z-direction
     * @param[in] it a teca_interval_iterator used to define the temporal layout
     *                                        of the files
     * @param[in] index_executive_compatability forces temporal_partion_size to 1
     * @param[in] index_request_key the name of the key to use when making requests
     *
     * @returns zero if successful
     */
    int initialize(MPI_Comm comm,
        unsigned long first_step, unsigned long last_step,
        unsigned long number_of_temporal_partitions,
        unsigned long temporal_partition_size, unsigned long *extent,
        unsigned long number_of_spatial_partitions,
        int partition_x, int partition_y, int partition_z,
        unsigned long min_block_size_x, unsigned long min_block_size_y,
        unsigned long min_block_size_z,
        const teca_calendar_util::p_interval_iterator &it,
        int index_executive_compatability, const std::string &index_request_key);

    int get_upstream_requests(teca_metadata base_req,
        std::vector<teca_metadata> &up_reqs) override;

    p_teca_cf_layout_manager get_layout_manager(long time_step) override;

    int get_layout_manager(const unsigned long temporal_extent[2],
        std::vector<p_teca_cf_layout_manager> &managers) override;

    int to_stream(std::ostream &os) override;

    void write_partitions();

protected:
    teca_cf_space_time_time_step_mapper() : file_steps()
    {}

    // remove these for convenience
    teca_cf_space_time_time_step_mapper(const teca_cf_space_time_time_step_mapper&) = delete;
    teca_cf_space_time_time_step_mapper(const teca_cf_space_time_time_step_mapper&&) = delete;
    void operator=(const teca_cf_space_time_time_step_mapper&) = delete;
    void operator=(const teca_cf_space_time_time_step_mapper&&) = delete;

    /// given a time step, get the corresponding file id
    int get_file_id(long time_step, long &file_id);

protected:
    using step_bracket_t = std::pair<long, long>;
    using temporal_extent_t = teca_coordinate_util::temporal_extent_t;
    using spatial_extent_t = teca_coordinate_util::spatial_extent_t;

    int index_executive_compatability;
    std::string index_request_key;
    std::vector<step_bracket_t> file_steps;
    spatial_extent_t whole_extent;
    std::vector<temporal_extent_t> temporal_partitions;
    std::deque<spatial_extent_t> spatial_partitions;
};

#endif
