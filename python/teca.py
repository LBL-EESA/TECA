try:
    # prevent mpi4py calling MPI_Init_threads during import. this is works
    # around the crash in MPI_Init_threads when not using srun on Cori. this
    # requires users of the module to explicitly call MPI_Init_threads and
    # MPI_Finalize.
    import mpi4py
    mpi4py.rc(initialize=False, finalize=False)

except ImportError:
    # mpi4py is not found
    pass

# bring in TECA. This includes all the wrapped C++ code
from teca_py import *
