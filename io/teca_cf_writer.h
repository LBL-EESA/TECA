#ifndef teca_cf_writer_h
#define teca_cf_writer_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_threaded_algorithm.h"
#include "teca_metadata.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cf_writer)

/// A writer for Cartesian meshes in NetCDF CF2 format.
/**
 * Writes data to NetCDF CF2 format. This algorithm is conceptually an
 * execution engine capable of driving the above pipeline with our without
 * threads and stream results in the order that they are generated placing them
 * in the correct location in the output dataset. The output dataset is a
 * collection of files each with a user specified number of time steps per
 * file. The output dataset may be arranged using a fixed number of steps per
 * file or daily, monthly, seasonal, or yearly file layouts. The total number
 * of time steps in the output dataset is determined by the combination of the
 * number of time steps in the input dataset and user defined subsetting if
 * any. The writer uses MPI collective I/O to produce the files. In parallel
 * time steps are mapped to ranks such that each rank has approximately the
 * same number of time steps. Incoming steps are mapped to files. A given MPI
 * rank may be writing to multiple files. The use of MPI collectives implies
 * care must be taken in its use to avoid deadlocks.
 *
 * Due to the use of MPI collectives I/O certain information must be known
 * during the report phase of pipeline execution, before the execute phase of
 * pipeline execution begins. The information that is needed is:
 *
 * ### number of time steps ###
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 * specified by the pipeline control index_initializer key found in metadata
 * produced by the source (e.g CF reader)
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * ### extent ###
 * ~~~~~~~~~~~~~~
 * 6, 64 bit integers defining the 3 spatial dimensions of each timestep found
 * in metadata produced by the source (e.g CF reader)
 * ~~~~~~~~~~~~~~
 *
 * ### point arrays ###
 * ~~~~~~~~~~~~~~~~~~~~
 * list of strings naming the point centered arrays that will be written. set
 * by the user prior to execution by writer properties.
 * ~~~~~~~~~~~~~~~~~~~~
 *
 * ### information arrays ###
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~
 * list of strings naming the non-geometric arrays that will written. set by
 * the user prior to execution by writer properties. See also size attribute
 * below.
 * ~~~~~~~~~~~~~~~~~~~~~~~~~~
 *
 * ### type_code ###
 * ~~~~~~~~~~~~~~~~~
 * the teca_variant_array_code naming the type of each array. this will be in
 * the array attributes metadata generated by the producer of the array (e.g
 * any algorithm that adds an array should provide this metadata).
 * ~~~~~~~~~~~~~~~~~
 *
 * ### size ###
 * ~~~~~~~~~~~~
 * a 64 bit integer declaring the size of each information array. this will be
 * in the array attributes metadata generated by the producer of the array (e.g
 * any algorithm that adds an array should provide this metadata).
 * ~~~~~~~~~~~~
 */
class TECA_EXPORT teca_cf_writer : public teca_threaded_algorithm
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

    /** @name file_name
     * Set the output filename. For time series the substring %t% is replaced
     * with the current time step or date. See comments on date_format below
     * for info about date formatting.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, file_name)
    ///@}


    /** @name date_format
     * set the format for the date to write in the filename. this requires the
     * input dataset to have unit/calendar information if none are available,
     * the time index is used instead. (%F-%HZ)
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(std::string, date_format)
    ///@}

    /** @name first_step
     * Set the first step in the range of time step to process.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, first_step)
    ///@}

    /** @name last_step
     * Set the last step in the range of time step to process.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, last_step)
    ///@}

    /** @name layout
     * Set the layout mode to one of : number_of_steps, daily, monthly,
     * seasonal, or yearly. This controls the size of the files written.  In
     * daily, monthly, seasonal, and yearly modes each file will contain the
     * steps spanning the given duration. The number_of_steps mode writes a
     * fixed number of steps per file which can be set using the
     * steps_per_file property.
     */
    ///@{
    enum {invalid=0, number_of_steps=1, daily=2, monthly=3, seasonal=4, yearly=5};
    TECA_ALGORITHM_PROPERTY_V(int, layout)

    void set_layout_to_number_of_steps() { this->set_layout(number_of_steps); }
    void set_layout_to_daily() { this->set_layout(daily); }
    void set_layout_to_monthly() { this->set_layout(monthly); }
    void set_layout_to_seasonal() { this->set_layout(seasonal); }
    void set_layout_to_yearly() { this->set_layout(yearly); }

    /// set the layout mode from a string.
    int set_layout(const std::string &layout);

    /// @returns 0 if the passed value is a valid layout mode
    int validate_layout(int mode)
    {
        if ((mode == number_of_steps) || (mode == daily) ||
             (mode == monthly) || (mode == seasonal) || (mode == yearly))
            return 0;

        TECA_ERROR("Invalid layout mode " << mode)
        return -1;
    }

    /// @returns a string representation of the current layout
    const char *get_layout_name() const;
    ///@}

    /** @name steps_per_file
     * Set how many time steps are written to each file when the layout mode is
     * set to number_of_steps.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(unsigned int, steps_per_file)
    ///@}

    /** @name mode_flags
     * sets the flags passed to NetCDF during file creation. (NC_CLOBBER)
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, mode_flags)
    ///@}

    /** @name use_unlimited_dim
     * if set the slowest varying dimension is specified to be NC_UNLIMITED.
     * This has a negative impact on performance when reading the values in a
     * single pass. However, unlimited dimensions are used ubiquitously thus
     * by default it is set. For data being consumed by TECA performance will
     * be better when using fixed dimensions. (1) This feature requires
     * collective writes and is incompatible with out of order execution,
     * and hence currently not supported.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, use_unlimited_dim)
    ///@}

    /** @name compression_level
     * sets the compression level used for each variable compression is not
     * used if the value is less than or equal to 0. This feature requires
     * collective writes and is incompatible with out of order execution,
     * and hence currently not supported.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, compression_level)
    ///@}

    /** @name collective_buffer
     * Enables MPI I/O colective buffering. Collective buffering is only valid
     * when the spatial partitioner is enabled and the number of spatial
     * partitions is equal to  the number of MPI ranks. If set to -1 (the
     * default) collective buffering will automatically enabled when it is
     * possible to do so.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, collective_buffer)
    ///@}

    /** @name flush_files
     * Flush files before closing them, this may be necessary if accessing data
     * immediately.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, flush_files)
    ///@}
;

    /** @name point_array
     * Specify the arrays to write. A data array is only written to disk if
     * it is included in this list. It is an error to not specify at least
     * one point centered array to write
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, point_array)
    ///@}

    /** @name information_array
     * Set the list of non-geometric arrays to write.
     */
    ///@{
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, information_array)
    ///@}

    /** @name spatial_partitioner
     * Enable spatial partitioner, When spatial partitioner is enabled both
     * temporal and spatial dimensions of input data are partitioned for load
     * balancing. The partitioner is controled by the
     * number_of_spatial_partitions, number_of_temporal_partitions and,
     * temporal_partition_size properties.
     */
    ///@{
    enum
    {
        temporal,  ///< map time steps to MPI ranks
        spatial,   ///< map spatial extents to MPI ranks, time is processed sequentially
        space_time ///< both spatial and temporal extents to MPI ranks
    };

    TECA_ALGORITHM_PROPERTY_V(int, partitioner)

    /// set the partitioner from a string
    void set_partitioner(const std::string &part);

    /// enables temporal partitioner
    void set_partitioner_to_temporal() { this->set_partitioner(temporal); }

    /// enables spatial partitioner
    void set_partitioner_to_spatial() { this->set_partitioner(spatial); }

    /// enables space-time partitioner
    void set_partitioner_to_space_time() { this->set_partitioner(space_time); }

    /// @returns 0 if the passed value is a valid partitioner mode
    int validate_partitioner(int mode)
    {
        if ((mode == temporal) || (mode == spatial) || (mode ==space_time))
            return 0;

        TECA_ERROR("Invalid partitioner mode " << mode)
        return -1;
    }

    /// @returns the name of the current partitioner mode
    const char *get_partitioner_name() const;
    ///@}

    /** @name number_of_spatial_partitions
     * Set the number of spatial partitions. If less than one then the number of
     * MPI ranks is used.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, number_of_spatial_partitions)
    ///@}

    /** @name partition_x
     * enables/disables spatial partitioning in the x-direction
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, partition_x)
    ///@}

    /** @name partition_y
     * enables/disables spatial partitioning in the y-direction
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, partition_y)
    ///@}

    /** @name partition_z
     * enables/disables spatial partitioning in the z-direction
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, partition_z)
    ///@}

    /** @name minimum_block_size_x
     * Sets the minimum block size for spatial partitioning in the x-direction
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, minimum_block_size_x)
    ///@}

    /** @name minimum_block_size_y
     * Sets the minimum block size for spatial partitioning in the y-direction
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, minimum_block_size_y)
    ///@}

    /** @name minimum_block_size_z
     * Sets the minimum block size for spatial partitioning in the z-direction
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, minimum_block_size_z)
    ///@}
    //
    /** @name number_of_temporal_partitions
     * Set the number of temporal partitions. If set to less than one then the
     * number of time steps is used. The temporal_partition_size property takes
     * precedence, if it is set then the this property is ignored. The default
     * value is zero.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, number_of_temporal_partitions)
    ///@}

    /** @name temporal_partition_size
     * Set the size of the temporal partitions. If set to less than one then the
     * number_of_temporal_partition property is used instead. The default value is
     * zero.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(long, temporal_partition_size)
    ///@}

    /** @name index_executive_compatibility
     * If set and spatial partitioner is enabled, the writer will make one
     * request per time step using the index_request_key as the
     * teca_index_executive would. This could be used parallelize existing
     * algorithms over space and time.
     */
    ///@{
    TECA_ALGORITHM_PROPERTY(int, index_executive_compatability)
    ///@}

protected:
    teca_cf_writer();

private:
    using teca_algorithm::get_output_metadata;
    using teca_algorithm::execute;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request, int streaming) override;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    std::vector<teca_metadata> get_upstream_request(unsigned int port,
        const std::vector<teca_metadata> &input_md,
        const teca_metadata &request) override;

    // flush data to disk. this may be necessary if accessing data
    // immediately.
    int flush();

private:
    std::string file_name;
    std::string date_format;
    long number_of_spatial_partitions;
    int partition_x;
    int partition_y;
    int partition_z;
    long minimum_block_size_x;
    long minimum_block_size_y;
    long minimum_block_size_z;
    long number_of_temporal_partitions;
    long temporal_partition_size;
    long first_step;
    long last_step;
    int layout;
    int partitioner;
    int index_executive_compatability;
    unsigned int steps_per_file;
    int mode_flags;
    int use_unlimited_dim;
    int collective_buffer;
    int compression_level;
    int flush_files;

    std::vector<std::string> point_arrays;
    std::vector<std::string> information_arrays;

    class internals_t;
    internals_t *internals;
};

#endif
