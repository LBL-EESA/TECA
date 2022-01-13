%{
#include "teca_config.h"
%}

%ignore TECA_VERSION_DESCR;
%ignore TECA_PYTHON_VERSION;
%include "teca_config.h"


%inline
%{
// return TECA version as a string
const char *get_teca_version_descr()
{
    return TECA_VERSION_DESCR;
}

// return version of Python TECA was compiled against
int get_teca_python_version()
{
    return TECA_PYTHON_VERSION;
}

// return true if the location of TECA_data repo is known
bool get_teca_has_data()
{
#if defined(TECA_HAS_DATA)
    return true;
#else
    return false;
#endif
}

// return the path to a TECA_data repo check out, or None if the
// path is not known at compile time
const char *get_teca_data_root()
{
#if defined(TECA_HAS_DATA)
    return TECA_DATA_ROOT;
#else
    return nullptr;
#endif
}

// return true if TECA was compiled with regex support
bool get_teca_has_regex()
{
#if defined(TECA_HAS_REGEX)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with netcdf support
bool get_teca_has_netcdf()
{
#if defined(TECA_HAS_NETCDF)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with MPI support
bool get_teca_has_mpi()
{
#if defined(TECA_HAS_MPI)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with boost support
bool get_teca_has_boost()
{
#if defined(TECA_HAS_BOOST)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with VTK support
bool get_teca_has_vtk()
{
#if defined(TECA_HAS_VTK)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with paraview support
bool get_teca_has_paraview()
{
#if defined(TECA_HAS_PARAVIEW)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with udunits support
bool get_teca_has_udunits()
{
#if defined(TECA_HAS_UDUNITS)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with OpenSSL support
bool get_teca_has_openssl()
{
#if defined(TECA_HAS_OPENSSL)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with NumPy support
bool get_teca_has_numpy()
{
#if defined(TECA_HAS_NUMPY)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with CuPy support
bool get_teca_has_cupy()
{
#if defined(TECA_HAS_CUPY)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with PyTorch support
bool get_teca_has_pytorch()
{
#if defined(TECA_HAS_PYTORCH)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with matplotlib support
bool get_teca_has_matplotlib()
{
#if defined(TECA_HAS_MATPLOTLIB)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with tcpyPI support
bool get_teca_has_tcpypi()
{
#if defined(TECA_HAS_TCPYPI)
    return true;
#else
    return false;
#endif
}

// return true if TECA was compiled with CUDA support
bool get_teca_has_cuda()
{
#if defined(TECA_HAS_CUDA)
    return true;
#else
    return false;
#endif
}
%}
