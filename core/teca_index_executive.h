#ifndef teca_index_executive_h
#define teca_index_executive_h

#include "teca_shared_object.h"
#include "teca_algorithm_executive.h"
#include "teca_metadata.h"
#include "teca_mpi.h"

#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_index_executive)

/// An executive that generates requests using a upstream or user defined index.
/** An extent or bounds to subset by, and list of arrays can be optionally set.
 *
 * metadata keys:
 *
 * requires:
 *
 *      index_initializer_key -- holds the name of the key that tells how
 *                               many indices are available. the named key
 *                               must also be present and should contain the
 *                               number of indices available
 *
 *      index_request_key -- holds the name of the key used to request
 *                           a specific index. request are generated with this
 *                           name set to a specific index to be processed some
 *                           upstream algorithm is expected to produce the
 *                           data associated with the given index
*/
class teca_index_executive : public teca_algorithm_executive
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

    /// @copydoc set_extent
    void set_extent(const std::vector<unsigned long> &ext);

    /** Set the bounds to process. If nothing is set then extent as provided by
     * set_extent is used.
     */
    void set_bounds(double *bounds);
    void set_bounds(const std::vector<double> &bounds);

    /// Set the list of arrays to process
    void set_arrays(const std::vector<std::string> &arrays);

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
};

#endif
