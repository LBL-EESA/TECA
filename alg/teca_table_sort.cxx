#include "teca_table_sort.h"

#include "teca_table.h"
#include "teca_array_collection.h"
#include "teca_variant_array.h"
#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_metadata.h"

#include <algorithm>
#include <iostream>
#include <string>
#include <set>
#include <cmath>

#if defined(TECA_HAS_BOOST)
#include <boost/program_options.hpp>
#endif

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

using std::string;
using std::vector;
using std::set;
using std::cerr;
using std::endl;
using namespace teca_variant_array_util;

//#define TECA_DEBUG

namespace internal
{
template<typename num_t>
class less
{
public:
    less() : m_data(nullptr) {}
    less(const num_t *data) : m_data(data) {}

    bool operator()(const size_t &l, const size_t &r)
    {
        return m_data[l] < m_data[r];
    }
private:
    const num_t *m_data;
};

template<typename num_t>
class greater
{
public:
    greater() : m_data(nullptr) {}
    greater(const num_t *data) : m_data(data) {}

    bool operator()(const size_t &l, const size_t &r)
    {
        return m_data[l] > m_data[r];
    }
private:
    const num_t *m_data;
};
};


// --------------------------------------------------------------------------
teca_table_sort::teca_table_sort() :
    index_column(""), index_column_id(0), stable_sort(0), ascending_order(0)
{
    this->set_number_of_input_connections(1);
    this->set_number_of_output_ports(1);
}

// --------------------------------------------------------------------------
teca_table_sort::~teca_table_sort()
{}

#if defined(TECA_HAS_BOOST)
// --------------------------------------------------------------------------
void teca_table_sort::get_properties_description(
    const string &prefix, options_description &global_opts)
{
    options_description opts("Options for "
        + (prefix.empty()?"teca_table_sort":prefix));

    opts.add_options()
        TECA_POPTS_GET(std::string, prefix, index_column,
            "name of the column to sort the table by")
        TECA_POPTS_GET(int, prefix, index_column_id,
            "column number to sort the table by. can be used in "
            "place of an index_column name")
        TECA_POPTS_GET(int, prefix, stable_sort,
            "if set a stable sort will be used")
        TECA_POPTS_GET(int, prefix, ascending_order,
            "if set the table is sorted in ascending order")
        ;

    this->teca_algorithm::get_properties_description(prefix, opts);

    global_opts.add(opts);
}

// --------------------------------------------------------------------------
void teca_table_sort::set_properties(
    const string &prefix, variables_map &opts)
{
    this->teca_algorithm::set_properties(prefix, opts);

    TECA_POPTS_SET(opts, std::string, prefix, index_column)
    TECA_POPTS_SET(opts, int, prefix, index_column_id)
    TECA_POPTS_SET(opts, int, prefix, stable_sort)
    TECA_POPTS_SET(opts, int, prefix, ascending_order)
}
#endif

// --------------------------------------------------------------------------
const_p_teca_dataset teca_table_sort::execute(
    unsigned int port,
    const std::vector<const_p_teca_dataset> &input_data,
    const teca_metadata &request)
{
#ifdef TECA_DEBUG
    cerr << teca_parallel_id() << "teca_table_sort::execute" << endl;
#endif
    (void)port;
    (void)request;

    // get the input
    const_p_teca_table in_table
        = std::dynamic_pointer_cast<const teca_table>(input_data[0]);

    // in parallel only rank 0 is required to have data
    int rank = 0;
#if defined(TECA_HAS_MPI)
    int init = 0;
    MPI_Initialized(&init);
    if (init)
        MPI_Comm_rank(this->get_communicator(), &rank);
#endif
    if (!in_table)
    {
        if (rank == 0)
        {
            TECA_FATAL_ERROR("empty input")
        }
        return nullptr;
    }

    // get the array to sort by
    const_p_teca_variant_array index_col;
    if (this->index_column.empty())
        index_col = in_table->get_column(this->index_column_id);
    else
        index_col = in_table->get_column(this->index_column);
    if (!index_col)
    {
        TECA_FATAL_ERROR("Failed to locate column to sort by \""
            <<  this->index_column << "\"")
        return nullptr;
    }

    // create the index
    unsigned long n_rows = index_col->size();
    unsigned long *index = static_cast<unsigned long*>(
        malloc(n_rows*sizeof(unsigned long)));
    for (unsigned long i = 0; i < n_rows; ++i)
        index[i] = i;

    VARIANT_ARRAY_DISPATCH(index_col.get(),

        auto [scol, col] = get_host_accessible<CTT>(index_col);

        sync_host_access_any(index_col);

        if (this->stable_sort)
        {
            if (this->ascending_order)
                std::stable_sort(index, index+n_rows, internal::greater<NT>(col));
            else
                std::stable_sort(index, index+n_rows, internal::less<NT>(col));
        }
        else
        {
            if (this->ascending_order)
                std::sort(index, index+n_rows, internal::greater<NT>(col));
            else
                std::sort(index, index+n_rows, internal::less<NT>(col));
        }
        )

    // transfer data and reorder
    p_teca_table out_table = teca_table::New();
    out_table->copy_metadata(in_table);
    out_table->copy_structure(in_table);
    unsigned int n_cols = out_table->get_number_of_columns();
    for (unsigned int j = 0; j < n_cols; ++j)
    {
        const_p_teca_variant_array in_col = in_table->get_column(j);
        p_teca_variant_array out_col = out_table->get_column(j);
        out_col->resize(n_rows);
        VARIANT_ARRAY_DISPATCH(out_col.get(),

            auto [sp_in_col, p_in_col] = get_host_accessible<CTT>(in_col);
            auto [p_out_col] = data<TT>(out_col);

            sync_host_access_any(in_col);

            for (unsigned long i = 0; i < n_rows; ++i)
                p_out_col[i] = p_in_col[index[i]];
            )
    }

    free(index);

    return out_table;
}
