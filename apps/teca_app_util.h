#ifndef teca_app_util_h
#define teca_app_util_h

/// @file

#include "teca_config.h"

#include <string>
#include <boost/program_options.hpp>

/// Codes shared among the command line applications
namespace teca_app_util
{

/** Check for flag and if found print the help message
 * and the option definitions. return non-zero if the flag
 * was found.
 */
int process_command_line_help(int rank, const std::string &flag,
    boost::program_options::options_description &opt_defs,
    boost::program_options::variables_map &opt_vals);

/** parses the command line options and checks for --help, --advanced_help, and
 * --full_help flags.  if any are found prints the associated option
 * defintions.  if any of the help flags were found 1 is returned. If there is
 * an error -1 is returned. Otherwise 0 is returned.
 */
int process_command_line_help(int rank, int argc, char **argv,
    boost::program_options::options_description &basic_opt_defs,
    boost::program_options::options_description &advanced_opt_defs,
    boost::program_options::options_description &all_opt_defs,
    boost::program_options::variables_map &opt_vals);

}

#endif
