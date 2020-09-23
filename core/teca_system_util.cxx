#include "teca_system_util.h"

namespace teca_system_util
{
// **************************************************************************
int get_environment_variable(const char *var, bool &val)
{
    const char *tmp = getenv(var);
    if (tmp)
    {
        char buf[17];
        buf[16] = '\0';
        size_t n = strlen(tmp);
        n = n < 17 ? n : 16;
        for (size_t i = 0; i < n && i < 16; ++i)
            buf[i] = tolower(tmp[i]);
        buf[n] = '\0';
        if ((strcmp(buf, "0") == 0)
            || (strcmp(buf, "false") == 0) || (strcmp(buf, "off") == 0))
        {
            val = false;
            return 0;
        }
        else if ((strcmp(buf, "1") == 0)
            || (strcmp(buf, "true") == 0) || (strcmp(buf, "on") == 0))
        {
            val = true;
            return 0;
        }
        else
        {
            TECA_ERROR("Failed to convert " << var << " = \""
                << tmp << "\" to a bool")
            return -1;
        }
    }
    return 1;
}
}
