#ifndef teca_cf_writer_h
#define teca_cf_writer_h

#include "teca_shared_object.h"
#include "teca_threaded_algorithm.h"
#include "teca_metadata.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cf_writer)

/**
an algorithm that writes cartesian meshes in NetCDF CF format.
*/
class teca_cf_writer : public teca_threaded_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cf_writer)
    ~teca_cf_writer();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the output filename. for time series the substring
    // %t% is replaced with the current time step.
    TECA_ALGORITHM_PROPERTY(std::string, file_name)

    // set how many time steps are written to each file. Note that upstream is
    // parallelized over files rather than time steps.  this has the affect of
    // reducing the available oportunity for MPI parallelization by this
    // factor. For example if there are 16 timee steps and steps_per_file is 8,
    // 2 MPI ranks each running 8 or more threads would be optimal. One
    // should make such calculations when planning large runs if optimal
    // performance is desired. time steps are gathered before the file is
    // written, thus available memory per MPI rank is the limiting factor in
    // how many steps can be stored in a single file.
    TECA_ALGORITHM_PROPERTY(unsigned int, steps_per_file)

    // sets the flags passed to NetCDF during file creation. (NC_CLOBBER)
    TECA_ALGORITHM_PROPERTY(int, mode_flags)

    // if set the slowest varying dimension is specified to be NC_UNLIMITED.
    // This has a negative impact on performance when reading the values in a
    // single pass. However, unlimited dimensions are used ubiquitously thus
    // by default it is set. For data being consumed by TECA performance will
    // be better when using fixed dimensions. (1)
    TECA_ALGORITHM_PROPERTY(int, use_unlimited_dim)

protected:
    teca_cf_writer();

private:
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;
private:
    std::string file_name;
    unsigned int steps_per_file;
    int mode_flags;
    int use_unlimited_dim;
};

#endif
