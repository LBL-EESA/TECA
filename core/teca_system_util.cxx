#include "teca_system_util.h"

#include <cstring>

namespace teca_system_util
{

// --------------------------------------------------------------------------
int get_command_line_option(int argc, char **argv,
    const char *arg_name, int require, std::string &arg_val)
{
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(arg_name, argv[i]) == 0)
        {
            if (++i == argc)
            {
                TECA_ERROR(<< arg_name << " is missing its value")
                return -1;
            }
            arg_val = argv[i];
            break;
        }
    }

    if (require && arg_val.empty())
    {
        TECA_ERROR("missing required command line option " << arg_name)
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int command_line_option_check(int argc, char **argv, const char *arg_name)
{
    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(arg_name, argv[i]) == 0)
            return 1;
    }
    return 0;
}

}
