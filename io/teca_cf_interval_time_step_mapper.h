#ifndef teca_cf_interval_time_step_mapper_h
#define teca_cf_interval_time_step_mapper_h

#include "teca_metadata.h"
#include "teca_cf_layout_manager.h"
#include "teca_cf_time_step_mapper.h"
#include "teca_calendar_util.h"
#include "teca_mpi.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

class teca_cf_interval_time_step_mapper;
using p_teca_cf_interval_time_step_mapper = std::shared_ptr<teca_cf_interval_time_step_mapper>;

/// NetCDF CF2 files time step mapper.
class teca_cf_interval_time_step_mapper : public teca_cf_time_step_mapper
{
public:

    // allocate and return a new object
    static p_teca_cf_interval_time_step_mapper New()
    { return p_teca_cf_interval_time_step_mapper(new teca_cf_interval_time_step_mapper); }

    ~teca_cf_interval_time_step_mapper() {}

    // initialize based on input metadata. this is a collective call
    // creates communicator groups for each file and creates the
    // file layout managers for the local rank. After this call
    // on can access file managers to create, define and write
    // local datasets to the NetCDF files in cf format.
    int initialize(MPI_Comm comm, long first_step, long last_step,
        const teca_calendar_util::p_interval_iterator &it,
        const teca_metadata &md);

    // given a time step, get the corresponding layout manager that
    // can be used to create, define and write data to disk.
    p_teca_cf_layout_manager get_layout_manager(long time_step) override;

    // print a summary to the stream
    int to_stream(std::ostream &os) override;

protected:
    teca_cf_interval_time_step_mapper() : file_steps()
    {}

    // remove these for convenience
    teca_cf_interval_time_step_mapper(const teca_cf_interval_time_step_mapper&) = delete;
    teca_cf_interval_time_step_mapper(const teca_cf_interval_time_step_mapper&&) = delete;
    void operator=(const teca_cf_interval_time_step_mapper&) = delete;
    void operator=(const teca_cf_interval_time_step_mapper&&) = delete;

    // given a time step, get the corresponding file id
    int get_file_id(long time_step, long &file_id);

protected:
    //using teca_calendar_util::time_point;
    //using step_bracket_t = std::pair<time_point, time_point>;
    using step_bracket_t = std::pair<long, long>;
    std::vector<step_bracket_t> file_steps;
};

#endif
