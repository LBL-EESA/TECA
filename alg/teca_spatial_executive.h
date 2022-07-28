#ifndef teca_spatial_executive_h
#define teca_spatial_executive_h

#include "teca_shared_object.h"
#include "teca_algorithm_executive.h"
#include "teca_metadata.h"
#include "teca_mpi.h"

#include <vector>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_spatial_executive)

/** An executive that generates requests such that the upstream or user defined
 * index space is partitoned using a fixed partition size and spatially across MPI
 * ranks. Set the partition size to 0 to use only the spatial partitioning.
 *
 * This executive uses the following meta data keys:
 *
 *     | Key  | Description |
 *     | ---- | ----------- |
 *     | whole_extent | The spatial extent to partiton. Partitionaing is |
 *     | ^            | applied such that each MPI rank has a unique subset to |
 *     | ^            | process |
 *     | index_initializer_key -| holds the name of the key that tells how |
 *     | ^                      | many indices are available. the named key |
 *     | ^                      | must also be present and should conatin the |
 *     | ^                      | number of indices available |
 *     | index_request_key | holds the name of the key used to request |
 *     | ^                 | a specific inclusive range of indices [i0, i1]. |
 *     | ^                 | Requests are generated with this name set to |
 *     | ^                 | a specific inclusive range of indices to be |
 *     | ^                 | processed. Some upstream algorithm is |
 *     | ^                 | expected to produce the data associated |
 *     | ^                 | with the requested index extent. |
 *
 */
class TECA_EXPORT teca_spatial_executive : public teca_algorithm_executive
{
public:
    TECA_ALGORITHM_EXECUTIVE_STATIC_NEW(teca_spatial_executive)

    int initialize(MPI_Comm comm, const teca_metadata &md) override;
    teca_metadata get_next_request() override;

    /** Set the number of partitions to partition the temporal extent into. Use
     * set_time_partition_size to control the partition size rather then the number.
     */
    void set_number_of_temporal_partitions(long n_partitions);

    /** Set the partition size to use when partitioning the temporal extent.
     * Setting to less than one results in a spatial only partitioning. When
     * greater or equal to one the value will be used to determine the number
     * of temporal partitions.
     *
     * When the total number of time steps is not evenly devided by the
     * temporal partition size, the final request will specify the remainder of
     * steps.
     *
     * @param[in] n_steps the number of indices to request per request.
     */
    void set_temporal_partition_size(long n_steps);

    /// Set the time step to process
    void set_time_step(long s);

    /// Set the first index in the series to process.The default is 0.
    void set_first_step(long s);

    /** set the last index in the series to process.  default is -1. negative
     * number results in the last available time step being used.
     */
    void set_last_step(long s);

    /** Set the extent to partition and process. The default is to partition
     * and process the whole_extent.
     */
    void set_extent(unsigned long *ext);
    void set_extent(const std::vector<unsigned long> &ext);

    /** Set the spatial bounds to partition and process. The default is to
     * partition and process the entire domain.
     */
    void set_bounds(double *bounds);
    void set_bounds(const std::vector<double> &bounds);

    /** Set a list of arrays to include in upstream requests. This provides a
     * means of pulling data through the pipeline when the upstream algorithms
     * do not do this automatically or to pull additional arrays.
     */
    void set_arrays(const std::vector<std::string> &arrays);

    /** Condigures the executive to make one request per time step using the
     * index_request_key as the teca_index_executive would. This could be used
     * parallelize exsiting algorithms over space and time.
     */
    void enable_index_executive_compatibility()
    {
        this->number_of_temporal_partitions = 0;
        this->temporal_partition_size = 1;
        this->index_executive_compatability = 1;
    }

    /// A flag that enables/disables partitioning in the x-direction
    void set_partition_x(int val) { this->partition_x = val; }

    /// A flag that enables/disables partitioning in the y-direction
    void set_partition_y(int val) { this->partition_y = val; }

    /// A flag that enables/disables partitioning in the z-direction
    void set_partition_z(int val) { this->partition_z = val; }

    /// Sets the minimum block size in the x-direction
    void set_minimum_block_size_x(unsigned long val)
    { this->minimum_block_size_x = std::max(1ul, val); }

    /// Sets the minimum block size in the y-direction
    void set_minimum_block_size_y(unsigned long val)
    { this->minimum_block_size_y = std::max(1ul, val); }

    /// Sets the minimum block size in the z-direction
    void set_minimum_block_size_z(unsigned long val)
    { this->minimum_block_size_z = std::max(1ul, val); }

protected:
    teca_spatial_executive();

private:
    std::vector<teca_metadata> requests;
    std::string index_initializer_key;
    std::string index_request_key;
    long first_step;
    long last_step;
    long number_of_temporal_partitions;
    long temporal_partition_size;
    int index_executive_compatability;
    int partition_x;
    int partition_y;
    int partition_z;
    unsigned long minimum_block_size_x;
    unsigned long minimum_block_size_y;
    unsigned long minimum_block_size_z;
    std::vector<unsigned long> extent;
    std::vector<double> bounds;
    std::vector<std::string> arrays;
};

#endif
