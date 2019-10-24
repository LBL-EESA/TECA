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
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cf_writer)
    TECA_ALGORITHM_CLASS_NAME(teca_cf_writer)
    ~teca_cf_writer();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // set the output filename. for time series the substring %t% is replaced
    // with the current time step or date. See comments on date_format below
    // for info about date formatting.
    TECA_ALGORITHM_PROPERTY(std::string, file_name)

    // set the format for the date to write in the filename this requires the
    // input dataset to have unit/calendar information if none are available,
    // the time index is used instead. (%F-%H)
    TECA_ALGORITHM_PROPERTY(std::string, date_format)

    // set the range of time step to process.
    TECA_ALGORITHM_PROPERTY(long, first_step)
    TECA_ALGORITHM_PROPERTY(long, last_step)

    // set how many time steps are written to each file. Note that upstream is
    // parallelized over files rather than time steps.  this has the affect of
    // reducing the available oportunity for MPI parallelization by this
    // factor. For example if there are 16 timee steps and steps_per_file is 8,
    // 2 MPI ranks each running 8 or more threads would be optimal. One
    // should make such calculations when planning large runs if optimal
    // performance is desired. time steps are gathered before the file is
    // written, thus available memory per MPI rank is the limiting factor in
    // how many steps can be stored in a single file (1).
    TECA_ALGORITHM_PROPERTY(unsigned int, steps_per_file)

    // sets the flags passed to NetCDF during file creation. (NC_CLOBBER)
    TECA_ALGORITHM_PROPERTY(int, mode_flags)

    // if set the slowest varying dimension is specified to be NC_UNLIMITED.
    // This has a negative impact on performance when reading the values in a
    // single pass. However, unlimited dimensions are used ubiquitously thus
    // by default it is set. For data being consumed by TECA performance will
    // be better when using fixed dimensions. (1)
    TECA_ALGORITHM_PROPERTY(int, use_unlimited_dim)


    // sets the compression level used for each variable
    // compression is not used if the value is less than
    // or equal to 0
    TECA_ALGORITHM_PROPERTY(int, compression_level)

protected:
    teca_cf_writer();

private:
    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request, int streaming) override;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;
private:
    std::string file_name;
    std::string date_format;
    long first_step;
    long last_step;
    unsigned int steps_per_file;
    int mode_flags;
    int use_unlimited_dim;
    int compression_level;

    class internals_t;
    internals_t *internals;
};

#endif
