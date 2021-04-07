#ifndef teca_cf_time_step_mapper_h
#define teca_cf_time_step_mapper_h

#include "teca_metadata.h"
#include "teca_cf_layout_manager.h"
#include "teca_mpi.h"

#include <iostream>
#include <sstream>
#include <cstring>
#include <cerrno>
#include <string>
#include <unordered_map>
#include <vector>
#include <memory>

class teca_cf_time_step_mapper;
using p_teca_cf_time_step_mapper = std::shared_ptr<teca_cf_time_step_mapper>;

/// NetCDF CF2 files time step mapper.
class teca_cf_time_step_mapper
{
public:

    // allocate and return a new object
    static p_teca_cf_time_step_mapper New()
    { return p_teca_cf_time_step_mapper(new teca_cf_time_step_mapper); }

    ~teca_cf_time_step_mapper() {}

    // initialize based on input metadata. this is a collective call
    // creates communicator groups for each file and creates the
    // file layout managers for the local rank. After this call
    // on can access file managers to create, define and write
    // local datasets to the NetCDF files in cf format.
    int initialize(MPI_Comm comm, long first_step, long last_step,
        long steps_per_file, const teca_metadata &md);

    // returns true if the mapper has been successfully initialized
    bool initialized() { return this->file_comms.size(); }

    // close all files, destroy file managers, and release communicators
    // this should be done once all I/O is complete.
    int finalize();

    // construct requests for this rank
    int get_upstream_requests(teca_metadata base_req,
        std::vector<teca_metadata> &up_reqs);

    // given a time step, get the corresponding layout manager that
    // can be used to create, define and write data to disk.
    p_teca_cf_layout_manager get_layout_manager(long time_step);

    // print a summary to the stream
    int to_stream(std::ostream &os);

    // call the passed in functor once per file table entry, safe
    // for MPI collective operations. The required functor signature
    // is:
    //      int f(long file_id, teca_cf_layout_manager &manager)
    //
    // a return of non-zero from the functor will immediately stop the
    // apply and the value will be returned, but no error will be
    // reported.
    template<typename op_t>
    int file_table_apply(const op_t &op);

protected:
    teca_cf_time_step_mapper() : index_initializer_key(""),
        index_request_key(""), start_time_step(0), end_time_step(-1),
        n_files(0), n_time_steps_per_file(1)
    {}

    // remove these for convenience
    teca_cf_time_step_mapper(const teca_cf_time_step_mapper&) = delete;
    teca_cf_time_step_mapper(const teca_cf_time_step_mapper&&) = delete;
    void operator=(const teca_cf_time_step_mapper&) = delete;
    void operator=(const teca_cf_time_step_mapper&&) = delete;

    // create/free the per-file communicators
    int alloc_file_comms();
    int free_file_comms();

    // given a time step, get the corresponding file id
    int get_file_id(long time_step, long &file_id);

private:
    // communicator to partition into per-file communicators
    MPI_Comm comm;

    // pipeline control key names
    std::string index_initializer_key;
    std::string index_request_key;

    // user provided overrides
    long start_time_step;
    long end_time_step;

    // time_steps to request by rank
    long n_time_steps;
    std::vector<long> block_size;
    std::vector<long> block_start;

    // output files
    long n_files;
    long n_time_steps_per_file;
    std::vector<std::set<int>> file_ranks;

    // per file communicators
    std::vector<MPI_Comm> file_comms;

    // the file table maps from a time step to a specific layout manager
    using file_table_t = std::unordered_map<long, p_teca_cf_layout_manager>;
    file_table_t file_table;
};


// --------------------------------------------------------------------------
template<typename op_t>
int teca_cf_time_step_mapper::file_table_apply(const op_t &op)
{
    for (long i = 0; i < this->n_files; ++i)
    {
        MPI_Comm comm_i = this->file_comms[i];
        if (comm_i != MPI_COMM_NULL)
        {
            if (int ierr = op(comm, i, this->file_table[i]))
                return ierr;
        }
    }
    return 0;
}

#endif
