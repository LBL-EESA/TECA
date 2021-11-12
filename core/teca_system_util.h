#ifndef teca_system_util_h
#define teca_system_util_h

/// @file

#include "teca_config.h"
#include "teca_common.h"
#include "teca_string_util.h"

#include <cstdlib>

/// Codes for dealing with low level system API's
namespace teca_system_util
{
/** initialize val with the environment variable named by var converted to a
 * numeric type. Only floating point and signed integers are implemented. For
 * unsigned types, check that the return is greater or equal to zero.
 *
 * returns:
 *    0  if the variable was found and val was initialized from it
 *    1  if the varibale was not found
 *   -1  if the variable was found but conversion from string failed
 */
template <typename T>
TECA_EXPORT
int get_environment_variable(const char *var, T &val)
{
    const char *tmp = getenv(var);
    if (tmp)
    {
        if (teca_string_util::string_tt<T>::convert(tmp, val))
        {
            TECA_ERROR("Failed to convert " << var << " = \""
                << tmp << "\" to a number")
            return -1;
        }
        return 0;
    }
    return 1;
}

/** extract the value of the named command line argument.  return 0 if
 * successful. If require is not zero then an error will be reported if the
 * argument is not present.
 */
TECA_EXPORT
int get_command_line_option(int argc, char **argv,
    const char *arg_name, int require, std::string &arg_val);

/** check for the presence of the name command line option.  return non-zero if
 * it is found.
 */
TECA_EXPORT
int command_line_option_check(int argc, char **argv,
    const char *arg_name);
}

#endif
