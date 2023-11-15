#include "teca_app_util.h"

#include "teca_config.h"
#include "teca_common.h"
#include "teca_file_util.h"
#include "teca_system_interface.h"

#include <exception>
#include <iostream>


namespace teca_app_util
{

// --------------------------------------------------------------------------
int process_command_line_help(int rank, const std::string &flag,
    boost::program_options::options_description &opt_defs,
    boost::program_options::variables_map &opt_vals)
{
    if (opt_vals.count(flag))
    {
        if (rank == 0)
        {
            std::string app_name =
                teca_file_util::filename(teca_system_interface::get_program_name());

            std::cerr << std::endl
                << "TECA version " << TECA_VERSION_DESCR
                << " compiled on " << __DATE__ << " " << __TIME__ << std::endl
                << std::endl
                << "Application usage: " << app_name << " [options]" << std::endl
                << std::endl
                << opt_defs << std::endl
                << std::endl;
        }
        return 1;
    }
    return 0;
}

// --------------------------------------------------------------------------
int process_command_line_help(int rank, int argc, char **argv,
    boost::program_options::options_description &basic_opt_defs,
    boost::program_options::options_description &advanced_opt_defs,
    boost::program_options::options_description &all_opt_defs,
    boost::program_options::variables_map &opt_vals)
{
    // this will prevent typos from being treated as positionals.
    boost::program_options::positional_options_description pos_opt_defs;

    try
    {
        boost::program_options::store(
            boost::program_options::command_line_parser(argc, argv)
                .style(boost::program_options::command_line_style::unix_style ^
                       boost::program_options::command_line_style::allow_short)
                .options(all_opt_defs)
                .positional(pos_opt_defs)
                .run(),
            opt_vals);

        if (process_command_line_help(rank, "help", basic_opt_defs, opt_vals) ||
            process_command_line_help(rank, "advanced_help", advanced_opt_defs, opt_vals) ||
            process_command_line_help(rank, "full_help", all_opt_defs, opt_vals))
        {
            return 1;
        }

        boost::program_options::notify(opt_vals);
    }
    catch (std::exception &e)
    {
        TECA_ERROR("Error parsing command line options. See --help "
            "for a list of supported options. " << e.what())
        return -1;
    }

    return 0;
}

}
