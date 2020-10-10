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

// return true if the location of TECA_assets repo is known
bool get_teca_has_assets()
{
#if defined(TECA_HAS_ASSETS)
    return true;
#else
    return false;
#endif
}

// return the path to a TECA_assets repo check out, or None if the
// path is not known at compile time
const char *get_teca_assets_root()
{
#if defined(TECA_HAS_ASSETS)
    return TECA_ASSETS_ROOT;
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
%}
