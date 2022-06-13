#ifndef teca_system_interface_h
#define teca_system_interface_h

/// @file

#include "teca_config.h"
#include "teca_mpi.h"
#include <string>

/// Codes for interfacing with low level system API's
namespace teca_system_interface
{

/// when set print stack trace in response to common signals.
TECA_EXPORT void set_stack_trace_on_error(int enable=1);

/// when set print stack trace in response to MPI errors.
TECA_EXPORT void set_stack_trace_on_mpi_error(MPI_Comm comm=MPI_COMM_WORLD, int enable=1);

/** return current program stack in a string demangle cxx symbols
 * if possible.
 *
 *    first_frame - frame number to start trace from
 *    whole_path -  set true to see full path in source file listing
*/
TECA_EXPORT std::string get_program_stack(int first_frame, int whole_path);

/// Return the name of the currently running program.
TECA_EXPORT std::string get_program_name();
};

#endif
