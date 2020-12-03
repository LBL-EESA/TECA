#include "teca_string_util.h"

namespace teca_string_util
{

// **************************************************************************
int extract_string(const char *istr, std::string &field)
{
    const char *sb = istr;
    while (*sb != '"')
    {
        if (*sb == '\0')
        {
            TECA_ERROR("End of string encountered before opening \"")
            return -1;
        }
        ++sb;
    }
    ++sb;
    const char *se = sb;
    while (*se != '"')
    {
        if (*se == '\\')
        {
            ++se;
        }
        if (*se == '\0')
        {
            TECA_ERROR("End of string encountered before closing \"")
            return -1;
        }
        ++se;
    }
    field = std::string(sb, se);
    return 0;
}

// **************************************************************************
int tokenize(char *istr, char delim, int n_cols, char **ostr)
{
    // skip delim at the beginning
    while ((*istr == delim) && (*istr != '\0'))
        ++istr;

    // nothing here
    if (*istr == '\0')
        return -1;

    // save the first
    ostr[0] = istr;
    int col = 1;

    while ((*istr != '\0') && (col < n_cols))
    {
        // seek to delim
        while ((*istr != delim) && (*istr != '\0'))
            ++istr;

        if (*istr == delim)
        {
            // terminate the token
            *istr = '\0';

            // move past the terminator
            ++istr;

            // check for end, if not start the next token
            if (*istr != '\0')
                ostr[col] = istr;

            // count it
            ++col;
        }
    }

    // we should have found n_cols
    if (col != n_cols)
    {
        TECA_ERROR("Failed to process all the data, "
            << col << "columns of the " << n_cols
            << " expected were processed.")
        return -1;
    }

    return 0;
}

}
