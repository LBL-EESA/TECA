#ifndef teca_mpi_manager_h
#define teca_mpi_manager_h

/// A RAII class to ease MPI initalization and finalization
// MPI_Init is handled in the constructor, MPI_Finalize is
// handled in the destructor.
class teca_mpi_manager
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
