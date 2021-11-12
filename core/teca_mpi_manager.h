#ifndef teca_mpi_manager_h
#define teca_mpi_manager_h

#include "teca_config.h"

/// A RAII class to ease MPI initalization and finalization
// MPI_Init is handled in the constructor, MPI_Finalize is handled in the
// destructor. Given that this is an application level helper rank and size
// are reported relative to MPI_COMM_WORLD.
class TECA_EXPORT teca_mpi_manager
{
public:
    teca_mpi_manager() = delete;
    teca_mpi_manager(const teca_mpi_manager &) = delete;
    void operator=(const teca_mpi_manager &) = delete;

    teca_mpi_manager(int &argc, char **&argv);
    ~teca_mpi_manager();

    int get_comm_rank(){ return m_rank; }
    int get_comm_size(){ return m_size; }

private:
    int m_rank;
    int m_size;
};

#endif
