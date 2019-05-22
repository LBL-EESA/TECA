#include "teca_table_reader.h"
#include "teca_table.h"
#include "teca_binary_stream.h"
#include "teca_coordinate_util.h"
#include "teca_file_util.h"

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <errno.h>

using std::string;
using std::endl;
using std::cerr;

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

// PIMPL idiom
struct teca_table_reader::teca_table_reader_internals
{
    teca_table_reader_internals()
        : number_of_indices(0) {}

    void clear();

    static p_teca_table read_table(const std::string &file_name,
        MPI_Comm comm, bool distribute);

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
teca_table_reader::teca_table_reader_internals::read_table(
    const std::string &file_name, MPI_Comm comm, bool distribute)
{
    teca_binary_stream stream;
#if !defined(TECA_HAS_MPI)
    (void)comm;
    (void)distribute;
#else
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
        if (teca_file_util::read_stream(file_name.c_str(),
            "teca_table", stream))
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
    table->from_stream(stream);
    return table;
}


// --------------------------------------------------------------------------
teca_table_reader::teca_table_reader() : generate_original_ids(0)
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
            "name of the column containing index values (\"\")")
        TECA_POPTS_GET(int, prefix, generate_original_ids,
            "add original row ids into the output. default off.")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, metadata_column_names,
             "names of the columns to copy directly into metadata")
        TECA_POPTS_MULTI_GET(std::vector<std::string>, prefix, metadata_column_keys,
             "names of the metadata keys to create from the named columns")
        ;

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_table_reader::set_properties(const string &prefix, variables_map &opts)
{
    TECA_POPTS_SET(opts, string, prefix, file_name)
    TECA_POPTS_SET(opts, string, prefix, index_column)
    TECA_POPTS_SET(opts, int, prefix, generate_original_ids)
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
            this->file_name, this->get_communicator(), distribute);

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
        TECA_ERROR("Table is missing the index array \""
            << this->index_column << "\"")
        return teca_metadata();
    }

    TEMPLATE_DISPATCH_I(teca_variant_array_impl,
        index.get(),

        NT *pindex = dynamic_cast<TT*>(index.get())->get();

        teca_coordinate_util::get_table_offsets(pindex,
            this->internals->table->get_number_of_rows(),
            this->internals->number_of_indices, this->internals->index_counts,
            this->internals->index_offsets, this->internals->index_ids);
        )

    // must have at least one index
    if (this->internals->number_of_indices < 1)
    {
        this->clear_cached_metadata();
        TECA_ERROR("Invalid index \"" << this->index_column << "\"")
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
                TECA_ERROR("metadata column \"" << md_col_name << "\" not found")
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
        TECA_ERROR("Failed to read data")
        return nullptr;
    }
#endif
    // get the requested index
    unsigned long index = 0;
    request.get("object_id", index);


    // not running in subsetting/parallel mode retrun
    // the complete table, it is empty off rank 0
    bool distribute = !this->index_column.empty();
    if (!distribute)
    {
        // copy the data, add the executive control keys
        if (this->internals->table)
        {
            p_teca_table out_table = teca_table::New();
            out_table->shallow_copy(this->internals->table);
            teca_metadata &md = out_table->get_metadata();
            md.set("index_request_key", std::string("object_id"));
            md.set("object_id", index);
            return out_table;
        }
        return nullptr;
    }

    // subset the table, pull out only rows for the requested index
    p_teca_table out_table = teca_table::New();

    out_table->copy_structure(this->internals->table);
    out_table->copy_metadata(this->internals->table);

    teca_metadata &md = out_table->get_metadata();
    md.set("index_request_key", std::string("object_id"));
    md.set("object_id", index);

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

        TEMPLATE_DISPATCH(teca_variant_array_impl,
            out_col.get(),
            NT *pin_col = static_cast<TT*>(in_col.get())->get();
            NT *pout_col = static_cast<TT*>(out_col.get())->get();
            memcpy(pout_col, pin_col+first_row, nrows*sizeof(NT));
            )
    }

    if (this->generate_original_ids)
    {
        p_teca_unsigned_long_array ids = teca_unsigned_long_array::New(nrows);
        unsigned long *pids = ids->get();
        for (unsigned long i = 0, q = first_row; i < nrows; ++i, ++q)
            pids[i] = q;
        out_table->append_column("original_ids", ids);
    }

    return out_table;
}
