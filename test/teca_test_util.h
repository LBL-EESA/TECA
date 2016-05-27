#ifndef teca_test_util_h
#define teca_test_util_h

#include "teca_config.h"
#include "teca_table.h"

#if defined(TECA_HAS_MPI)
#include <mpi.h>
#endif

namespace teca_test_util
{
// This creates a TECA table containing some basic test data that
// is used by the TECA table reader/writer tests and the dataset_diff
// test.
enum {base_table,
    break_string_col,
    break_int_col,
    break_float_col
    };
const_p_teca_table create_test_table(long step,
    int tid=teca_test_util::base_table);

#if defined(TECA_HAS_MPI)
// communication helper functions
template <typename num_t> struct mpi_tt {};
#define declare_mpi_tt(_ctype, _mpi_type)               \
template <> struct mpi_tt<_ctype>                       \
{static MPI_Datatype type_code() { return _mpi_type; } };
declare_mpi_tt(char, MPI_CHAR)
declare_mpi_tt(int, MPI_INT)
declare_mpi_tt(unsigned int, MPI_UNSIGNED)
declare_mpi_tt(long, MPI_LONG)
declare_mpi_tt(unsigned long, MPI_UNSIGNED_LONG)
declare_mpi_tt(float, MPI_FLOAT)
declare_mpi_tt(double, MPI_DOUBLE)
#endif

// **************************************************************************
template <typename num_t>
int bcast(num_t *buffer, size_t size)
{
#if defined(TECA_HAS_MPI)
    return MPI_Bcast(buffer, size, mpi_tt<num_t>::type_code(),
         0, MPI_COMM_WORLD);
#else
    (void)buffer;
    (void)size;
    return 0;
#endif
}

// **************************************************************************
template <typename num_t>
int bcast(num_t &buffer)
{
    return teca_test_util::bcast(&buffer, 1);
}

// **************************************************************************
int bcast(std::string &str);

// **************************************************************************
template <typename vec_t>
int bcast(std::vector<vec_t> &vec)
{
#if defined(TECA_HAS_MPI)
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    long vec_size = vec.size();
    if (teca_test_util::bcast(vec_size))
        return -1;
    if (rank != 0)
        vec.resize(vec_size);
    for (long i = 0; i < vec_size; ++i)
    {
        if (teca_test_util::bcast(vec[i]))
            return -1;
    }
#else
    (void)vec;
#endif
    return 0;
}

}
#endif
