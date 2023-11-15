#ifndef teca_cf_layout_manager_h
#define teca_cf_layout_manager_h

#include "teca_config.h"
#include "teca_metadata.h"
#include "teca_variant_array.h"
#include "teca_array_collection.h"
#include "teca_netcdf_util.h"
#include "teca_mpi.h"

#include <memory>
#include <vector>
#include <string>
#include <array>

class teca_cf_layout_manager;
using p_teca_cf_layout_manager = std::shared_ptr<teca_cf_layout_manager>;

/// Puts data on disk using NetCDF CF2 conventions.
class TECA_EXPORT teca_cf_layout_manager
{
public:
    // allocate and return a new object. The communicator passed in will be
    // used for collective operations, file_id uniquely identifies the file,
    // first_index and n_indices describe the index set that will be written to
    // the file by all ranks.
    static p_teca_cf_layout_manager New(MPI_Comm comm,
        long file_id, long first_index, long n_indices)
    {
        return p_teca_cf_layout_manager(
            new teca_cf_layout_manager(comm, file_id,
                first_index, n_indices));
    }

    /// creates the NetCDF file. This is an MPI collective call.
    int create(const std::string &file_name, const std::string &date_format,
        const teca_metadata &md_in, int mode_flags, int use_unlimited_dim);

    /** defines the NetCDF file layout. This is an MPI collective call. The
     * metadata object must contain global view of coordinates, whole_extent,
     * and for each array to be written there must be type code in the
     * corresponding array attributes.
     *
     * @param[in] md a metadata object compatible with that provided by the
     *               teca_cf_reader
     * @param[in] whole_extent the extent of data that will be written to disk,
     *                         including runtime specified subsetting.
     * @param[in] point_arrays a list of point centered array names to write
     * @param[in] info_arrays a list of point centered array names to write
     * @param[in] collective_buffer set to zero to disable collective buffering
     * @param[in] compression_level set greater than 1 to enable compression.
     *            this is incomatible with MPI parallel I/O and cannot be used
     *            in a parallel setting.
     *
     * @returns zero if successful
     */
    int define(const teca_metadata &md, unsigned long *whole_extent,
        const std::vector<std::string> &point_arrays,
        const std::vector<std::string> &info_arrays,
        int collective_buffer, int compression_level);

    /// writes the collection of arrays to the NetCDF file in the correct spot.
    int write(long index,
        const const_p_teca_array_collection &point_arrays,
        const const_p_teca_array_collection &info_arrays);

    /** Writes the collection of arrays defined over a spatio-temporal extent
     * to the NetCDF file in the correct location in the file.
     *
     * @param[in] extent       the spatial extent of the arrays
     * @param[in] temporal_extent the temporal extent of the arrays
     * @param[in] point_arrays a collection of point centered data arrays to
     *                         write
     * @param[in] info_arrays  a collection of non-geometrically oriented arrays
     *                         to write
     *
     * @returns zero if successful
     */
    int write(const unsigned long extent[6],
        const unsigned long temporal_extent[2],
        const const_p_teca_array_collection &point_arrays,
        const const_p_teca_array_collection &info_arrays);

    // close the file. This is an MPI collective call.
    int close()  { return this->handle.close(); }

    // return true if the file is open and can be written to
    bool opened()  { return bool(this->handle); }

    // return true if the file has been defined
    bool defined()  { return this->n_dims > 0; }

    // TODO -- this is only true when a rank writes all of the steps
    // to the given file.
    bool completed()
    {
        return this->n_written == this->n_indices;
    }

    // flush data to disk
    int flush();

    // print a summary to the stream
    int to_stream(std::ostream &os);

protected:
    teca_cf_layout_manager() : comm(MPI_COMM_SELF), file_id(-1),
        first_index(-1), n_indices(-1), n_written(0), n_dims(0),
        dims{0}
    {}

    teca_cf_layout_manager(MPI_Comm fcomm,
        long fid, long first_id, long n_ids) : comm(fcomm), file_id(fid),
        first_index(first_id), n_indices(n_ids), n_written(0), n_dims(0),
        dims{0}
    {}

    // remove these for now for convenience
    teca_cf_layout_manager(const teca_cf_layout_manager&) = delete;
    teca_cf_layout_manager(const teca_cf_layout_manager&&) = delete;
    void operator=(const teca_cf_layout_manager&) = delete;
    void operator=(const teca_cf_layout_manager&&) = delete;

protected:
    // communicator describing ranks that act on the file
    MPI_Comm comm;

    // identifying the file
    long file_id;
    std::string file_name;
    teca_netcdf_util::netcdf_handle handle;

    // for identifying the incoming dataset and determining its
    // position in the file
    long first_index;
    long n_indices;
    long n_written;

    // for low level NetCDF book keeping
    int mode_flags;
    int use_unlimited_dim;
    int n_dims;
    size_t dims[4];
    int mesh_axis[4];
    unsigned long whole_extent[6];

    struct var_def_t
    {
        var_def_t() : var_id(0), type_code(0), active_dims{0,0,0,0} {}

        var_def_t(int aid, unsigned int atc, const std::array<int,4> &ada) :
            var_id(aid), type_code(atc), active_dims(ada) {}

        var_def_t(int aid, unsigned int atc) :
            var_id(aid), type_code(atc), active_dims{0,0,0,0} {}

        int var_id;
        unsigned int type_code;
        std::array<int,4> active_dims;
    };

    std::map<std::string, var_def_t> var_def;
    std::string t_variable;
    p_teca_variant_array t;
};

#endif
