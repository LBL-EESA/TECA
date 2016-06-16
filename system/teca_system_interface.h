#ifndef teca_system_interface_h
#define teca_system_interface_h

#include <string>

namespace teca_system_interface
{
/**
when set print stack trace in response to common signals.
*/
void set_stack_trace_on_error(int enable=1);

/**
return current program stack in a string demangle cxx symbols
if possible.

first_frame - frame number to start trace from
whole_path -  set true to see full path in source file listing
*/
std::string get_program_stack(int first_frame, int whole_path);
};

#endif
