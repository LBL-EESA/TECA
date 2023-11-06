#ifndef teca_index_executive_h
#define teca_index_executive_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm_executive.h"
#include "teca_metadata.h"
#include "teca_mpi.h"

#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_index_executive)

/// An executive that generates requests using a upstream or user defined index.
/** An extent or bounds to subset by, and list of arrays can be optionally set.
 * This executive partitions an index set approximately equally accross the
 * available MPI ranks. Each rank is assigned a unique set of CUDA devices if
 * CUDA devices are available.  Within each rank requests are issued to the
 * assigned CUDA devices by setting the device_id key in the request. Upstream
 * algorithms should examine the device_id key and use the given device for
 * calculaitons. A device_id of -1 indicates that the CPU should be used for
 * calculations. Algorithms that do not have a CUDA implementation will make
 * use of the CPU and ignore the device_id field.
 *
 * ### Metadata keys:
 *
 * #### Requires:
 *
 * | Key                    | Description |
 * | ---------------------- | ----------- |
 * | index_initializer_key  | holds the name of the key that tells how many |
 * |                        | indices are available. the named key must also be |
 * |                        | present and should contain the number of indices |
 * |                        | available |
 * | index_request_key      | holds the name of the key used to request a |
 * |                        | specific index. request are generated with this |
 * |                        | name set to a specific index to be processed some |
 * |                        | upstream algorithm is expected to produce the |
 * |                        | data associated with the given index |
 *
 * #### Exports:
 *
 * | Key                    | Description |
 * | ---------------------- | ----------- |
 * | index_request_key      | The name of the key holding the requested index |
 * | <index_request_key>    | the requested index |
 * | device_id              | the CPU (-1) or CUDA device (0 - n-1 devices) to |
 * |                        | use for calculations |
 * | bounds                 | the [x0 x1 y0 y1 z0 z1] spatial bounds requested |
 * |                        | (optional) |
 * | extent                 | the [i0 i1 j0 j1 k0 k1] index space grid extent |
 * |                        | requested (optional) |
 * | arrays                 | a list of arrays requested (optional) |
 *
 */
class TECA_EXPORT teca_index_executive : public teca_algorithm_executive
{
public:
    TECA_ALGORITHM_EXECUTIVE_STATIC_NEW(teca_index_executive)

    int initialize(MPI_Comm comm, const teca_metadata &md) override;
    teca_metadata get_next_request() override;

    /// set the index to process
    void set_index(long s);

    // Set the first time step in the series to process. The default is 0.
    void set_start_index(long s);

    /** Set the last time step in the series to process.  default is -1.
     * negative number results in the last available time step being used.
     */
    void set_end_index(long s);

    /// Set the stride to process time steps at. The default is 1
    void set_stride(long s);

    /// Set the extent to process. The default is taken from whole_extent key.
    void set_extent(unsigned long *ext);

    /// @copydoc teca_index_executive::set_extent(unsigned long *)
    void set_extent(const std::vector<unsigned long> &ext);

    /** Set the bounds to process. If nothing is set then extent as provided by
     * ::set_extent is used.
     */
    void set_bounds(double *bounds);
    void set_bounds(const std::vector<double> &bounds);

    /// Set the list of arrays to process
    void set_arrays(const std::vector<std::string> &arrays);

    /// Set the list of devices to assign work to
    void set_device_ids(const std::vector<int> &device_ids);

protected:
    teca_index_executive();

private:
    std::vector<teca_metadata> requests;
    std::string index_initializer_key;
    std::string index_request_key;
    long start_index;
    long end_index;
    long stride;
    std::vector<unsigned long> extent;
    std::vector<double> bounds;
    std::vector<std::string> arrays;
    std::vector<int> device_ids;
};

#endif
