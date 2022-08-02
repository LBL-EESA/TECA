#ifndef teca_cf_block_time_step_mapper_h
#define teca_cf_block_time_step_mapper_h

#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_cf_layout_manager.h"
#include "teca_cf_time_step_mapper.h"
#include "teca_mpi.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

class teca_cf_block_time_step_mapper;
using p_teca_cf_block_time_step_mapper = std::shared_ptr<teca_cf_block_time_step_mapper>;

/// Maps time steps to files in fixed sized blocks
class TECA_EXPORT teca_cf_block_time_step_mapper : public teca_cf_time_step_mapper
{
public:

    /// allocate and return a new object
    static p_teca_cf_block_time_step_mapper New()
    { return p_teca_cf_block_time_step_mapper(new teca_cf_block_time_step_mapper); }

    ~teca_cf_block_time_step_mapper() {}

    /** initialize based on input metadata. this is a collective call creates
     * communicator groups for each file and creates the file layout managers
     * for the local rank. After this call on can access file managers to
     * create, define and write local datasets to the NetCDF files in cf
     * format.
     */
    int initialize(MPI_Comm comm, long first_step, long last_step,
        long steps_per_file, const std::string &index_request_key);

    p_teca_cf_layout_manager get_layout_manager(long time_step) override;
    using teca_cf_time_step_mapper::get_layout_manager;

    /// print a summary to the stream
    int to_stream(std::ostream &os) override;

protected:
    teca_cf_block_time_step_mapper() : n_time_steps_per_file(1)
    {}

    // remove these for convenience
    teca_cf_block_time_step_mapper(const teca_cf_block_time_step_mapper&) = delete;
    teca_cf_block_time_step_mapper(const teca_cf_block_time_step_mapper&&) = delete;
    void operator=(const teca_cf_block_time_step_mapper&) = delete;
    void operator=(const teca_cf_block_time_step_mapper&&) = delete;

    // given a time step, get the corresponding file id
    int get_file_id(long time_step, long &file_id);

protected:
    long n_time_steps_per_file;
};

#endif
