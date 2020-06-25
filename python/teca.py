# let the user disable MPI_Init. this is primarilly to work around
# Cray's practice of calling abort in MPI_Init on the login nodes.
import os, sys
try:
    tmp = os.environ['TECA_INITIALIZE_MPI']
    ltmp = tmp.lower()
    if (ltmp == 'false') or (ltmp == '0') or (ltmp == 'off'):
        import mpi4py
        mpi4py.rc(initialize=False, finalize=False)
        sys.stderr.write('WARNING: TECA_INITIALIZE_MPI=%s MPI_Init '
                         'was not called.\n'%(tmp))
except Exception:
    pass

# bring in TECA. This includes all the wrapped C++ code
from teca_py import *
