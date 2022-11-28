#include "teca_table_reader.h"
#include "teca_table.h"
#include "teca_binary_stream.h"
#include "teca_coordinate_util.h"
#include "teca_file_util.h"
#include "teca_common.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <errno.h>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using std::string;
using std::endl;
using std::cerr;

using namespace teca_variant_array_util;

// PIMPL idiom
struct teca_table_reader::teca_table_reader_internals
{
    teca_table_reader_internals()
        : number_of_indices(0) {}

    void clear();

    static p_teca_table read_table(MPI_Comm comm,
        const std::string &file_name, int file_format,
        bool distribute);

    p_teca_table table;
    teca_metadata metadata;
    unsigned long number_of_indices;
    std::vector<unsigned long> index_counts;
    std::vector<unsigned long> index_offsets;
    std::vector<unsigned long> index_ids;
};

// --------------------------------------------------------------------------
void teca_table_reader::teca_table_reader_internals::clear()
{
    this->table = nullptr;
    this->number_of_indices = 0;
    this->metadata.clear();
    this->index_counts.clear();
    this->index_offsets.clear();
    this->index_ids.clear();
}

// --------------------------------------------------------------------------
p_teca_table
teca_table_reader::teca_table_reader_internals::read_table(MPI_Comm comm,
    const std::string &file_name, int file_format, bool distribute)
{
#if !defined(TECA_HAS_MPI)
    (void)comm;
    (void)distribute;
#endif

    teca_binary_stream stream;

    // dtermine which format we are to read from
    if (file_format == teca_table_reader::format_auto)
    {
        std::string ext = teca_file_util::extension(file_name);
        if (ext == "bin")
        {
            file_format = teca_table_reader::format_bin;
        }
        else if (ext == "csv")
        {
            file_format = teca_table_reader::format_csv;
        }
        else
        {
            TECA_ERROR("Failed to determine file format from extension \""
                << ext << "\"")
            return nullptr;
        }
    }

    // validate the format override
    if ((file_format != teca_table_reader::format_bin) &&
        (file_format != teca_table_reader::format_csv))
    {
        TECA_ERROR("Invalid file format code " << file_format)
        return nullptr;
    }

#if defined(TECA_HAS_MPI)
    int init = 0;
    int rank = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(comm, &rank);

    // rank 0 will read the data, must be rank 0 for the
    // case where using as a serial reader, but running in
    // parallel. rank 0 is assumed to have the data in the
    // serial table based algorithms.
    const int root_rank = 0;
    if (rank == root_rank)
    {
#endif
        // set the header for the binary format. this is done to detect
        // compatible binary streams. for the csv fomat there is no header
        const char *header = nullptr;
        if (file_format == teca_table_reader::format_bin)
            header = "teca_table";

        if (teca_file_util::read_stream(file_name.c_str(), header, stream))
        {
            TECA_ERROR("Failed to read teca_table from \""
                << file_name << "\"")
            return nullptr;

        }
#if defined(TECA_HAS_MPI)
        if (init && distribute)
            stream.broadcast(comm);
    }
    else
    if (init && distribute)
    {
        stream.broadcast(comm);
    }
    else
    {
        return nullptr;
    }
#endif
    // deserialize the binary rep
    p_teca_table table = teca_table::New();

    if (file_format == teca_table_reader::format_bin)
    {
        if (table->from_stream(stream))
        {
            TECA_ERROR("Failed to deserialize binary stream from file \""
                << file_name << "\"")
            return nullptr;
        }
    }
    else if (file_format == teca_table_reader::format_csv)
    {
        size_t n_bytes = stream.size();
        const char *p_data = (const char*)stream.get_data();
        std::istringstream cpp_stream(std::string(p_data, p_data + n_bytes));
        if (table->from_stream(cpp_stream))
        {
            TECA_ERROR("Failed to deserialize std::stream from file \""
                << file_name << "\"")
            return nullptr;
        }
    }
    else
    {
        TECA_ERROR("Invalid file format " << file_format)
        return nullptr;
    }

    return table;
}


// --------------------------------------------------------------------------
teca_table_reader::teca_table_reader() : generate_original_ids(0),
    file_format(format_auto)
{
    this->internals = new teca_table_reader_internals;
}

// --------------------------------------------------------------------------
teca_table_reader::~teca_table_reader()
{
    delete this->internals;
}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_table_reader::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_table_reader":prefix));

    opts.add_options()
        TECA_POPTS_GET(string, prefix, file_name,
            "a file name to read")
        TECA_POPTS_GET(string, prefix, index_column,
            "name of the column containing index values")
        TECA_POPTS_GET(int, prefix, generate_original_ids,
            "add original row ids into the output. default off.")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, metadata_column_names,
             "names of the columns to copy directly into metadata")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, metadata_column_keys,
             "names of the metadata keys to create from the named columns")
        TECA_POPTS_GET(int, prefix, file_format,
            "output file format enum, 0:csv, 1:bin, 2:xlsx, 3:auto."
            "if auto is used, format is deduced from file_name")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_table_reader::set_properties(const string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, string, prefix, file_name)
    TECA_POPTS_SET(opts, string, prefix, index_column)
    TECA_POPTS_SET(opts, int, prefix, generate_original_ids)
    TECA_POPTS_SET(opts, int, prefix, file_format)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, metadata_column_names)
    TECA_POPTS_SET(opts, std::vector<std::string>, prefix, metadata_column_keys)
}
#endif

// --------------------------------------------------------------------------
void teca_table_reader::set_modified()
{
    // clear cached metadata before forwarding on to
    // the base class.
    this->clear_cached_metadata();
    teca_algorithm::set_modified();
}

// --------------------------------------------------------------------------
void teca_table_reader::clear_cached_metadata()
{
    this->internals->clear();
}

// --------------------------------------------------------------------------
teca_metadata teca_table_reader::get_output_metadata(unsigned int port,
    const std::vector<teca_metadata> &input_md)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id()
        << "teca_cf_reader::get_output_metadata" << endl;
#endif
    (void)port;
    (void)input_md;

    // if result is cached use that
    if (this->internals->table)
        return this->internals->metadata;

    // read the data
    bool distribute = !this->index_column.empty();

    this->internals->table =
        teca_table_reader::teca_table_reader_internals::read_table(
            this->get_communicator(), this->file_name, this->file_format,
            distribute);

    // when no index column is specified  act like a serial reader
    if (!this->internals->table || !distribute)
    {
        teca_metadata &md = this->internals->metadata;
        md.set("index_initializer_key", std::string("number_of_tables"));
        md.set("index_request_key", std::string("table_id"));
        md.set("number_of_tables", 1);
        return md;
    }

    // build the data structures for random access
    p_teca_variant_array index =
        this->internals->table->get_column(this->index_column);

    if (!index)
    {
        this->clear_cached_metadata();
        TECA_FATAL_ERROR("Table is missing the index array \""
            << this->index_column << "\"")
        return teca_metadata();
    }

    VARIANT_ARRAY_DISPATCH_I(index.get(),

        auto [pindex] = data<CTT>(index);

        teca_coordinate_util::get_table_offsets(pindex,
            this->internals->table->get_number_of_rows(),
            this->internals->number_of_indices, this->internals->index_counts,
            this->internals->index_offsets, this->internals->index_ids);
        )

    // must have at least one index
    if (this->internals->number_of_indices < 1)
    {
        this->clear_cached_metadata();
        TECA_FATAL_ERROR("Invalid index \"" << this->index_column << "\"")
        return teca_metadata();
    }

    // provide the names of keys used by the executive, and
    // fill the initializer in with the number of objects available
    // note an object could be a storm track or simply a cell in
    // the table
    teca_metadata md;
    md.set("index_initializer_key", std::string("number_of_objects"));
    md.set("index_request_key", std::string("object_id"));
    md.set("number_of_objects", this->internals->number_of_indices);

    // optionally pass columns directly into metadata
    size_t n_metadata_columns = this->metadata_column_names.size();
    if (n_metadata_columns)
    {
        for (size_t i = 0; i < n_metadata_columns; ++i)
        {
            std::string md_col_name = this->metadata_column_names[i];
            p_teca_variant_array md_col = this->internals->table->get_column(md_col_name);
            if (!md_col)
            {
                TECA_FATAL_ERROR("metadata column \"" << md_col_name << "\" not found")
                continue;
            }
            md.set(this->metadata_column_keys[i], md_col);
        }
    }

    // cache it
    this->internals->metadata = md;

    return md;
}


// --------------------------------------------------------------------------
const_p_teca_dataset teca_table_reader::execute(unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
    (void)port;
    (void)input_data;

#if defined(TECA_HAS_MPI)
    int rank = 0;
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(this->get_communicator(), &rank);
    if ((rank == 0) && !this->internals->table)
    {
        TECA_FATAL_ERROR("Failed to read data")
        return nullptr;
    }
#endif
    // get the requested index
    unsigned long index = 0;
    request.get("object_id", index);

    // not running in subsetting/parallel mode return
    // the complete table, it is empty off rank 0
    bool distribute = !this->index_column.empty();
    if (!distribute)
    {
        // copy the data, add the executive control keys
        if (this->internals->table)
        {
            p_teca_table out_table = teca_table::New();
            out_table->shallow_copy(this->internals->table);
            out_table->set_request_index("table_id", index);
            return out_table;
        }
        return nullptr;
    }

    // subset the table, pull out only rows for the requested index
    p_teca_table out_table = teca_table::New();

    out_table->copy_structure(this->internals->table);
    out_table->copy_metadata(this->internals->table);
    out_table->set_request_index("object_id", index);

    int ncols = out_table->get_number_of_columns();
    unsigned long nrows = this->internals->index_counts[index];
    unsigned long first_row = this->internals->index_offsets[index];

    for (int j = 0; j < ncols; ++j)
    {
        p_teca_variant_array in_col =
            this->internals->table->get_column(j);

        p_teca_variant_array out_col =
            out_table->get_column(j);

        out_col->resize(nrows);

        VARIANT_ARRAY_DISPATCH(out_col.get(),

            auto [pin_col, pout_col] = data<TT>(in_col, out_col);

            memcpy(pout_col, pin_col+first_row, nrows*sizeof(NT));
            )
    }

    if (this->generate_original_ids)
    {
        auto [ids, pids] = ::New<teca_unsigned_long_array>(nrows);

        for (unsigned long i = 0, q = first_row; i < nrows; ++i, ++q)
            pids[i] = q;

        out_table->append_column("original_ids", ids);
    }

    return out_table;
}
