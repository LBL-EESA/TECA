#ifndef teca_string_util_h
#define teca_string_util_h

/// @file

#include "teca_config.h"
#include "teca_common.h"

#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <string>
#include <vector>
#include <set>

/// Codes for dealing with string processing
namespace teca_string_util
{
/** Convert the characters between the first and second double
 * quote to a std::string. Escaped characters are skipped. Return
 * 0 if successful.
 */
TECA_EXPORT
int extract_string(const char *istr, std::string &field);

/** Scan the input string (istr) for the given a delimiter (delim). push a pointer
 * to the first non-delimiter character and the first character after each
 * instance of the delimiter.  return zero if successful. when successful there
 * will be at least one value.
 */
TECA_EXPORT
int tokenize(char *istr, char delim, int n_cols, char **ostr);


/** Scan the input string (istr) for the given a delimiter (delim). push a point
 * to the first non-delimiter character and the first character after each
 * instance of the delimiter.  return zero if successful. when successful there
 * will be at least one value.
 */
template <typename container_t = std::vector<char*>>
TECA_EXPORT
int tokenize(char *istr, char delim, container_t &ostr)
{
    // skip delim at the beginning
    while ((*istr == delim) && (*istr != '\0'))
        ++istr;

    // nothing here
    if (*istr == '\0')
        return -1;

    // save the first
    ostr.push_back(istr);

    while (*istr != '\0')
    {
        while ((*istr != delim) && (*istr != '\0'))
            ++istr;

        if (*istr == delim)
        {
            // terminate the token
            *istr = '\0';
            ++istr;
            if (*istr != '\0')
            {
                // not at the end, start the next token
                ostr.push_back(istr);
            }
        }
    }

    return 0;
}

/** Skip space, tabs, and new lines.  return non-zero if the end of the string
 * is reached before a non-pad character is encountered
 */
inline
int skip_pad(char *&buf)
{
    while ((*buf != '\0') &&
        ((*buf == ' ') || (*buf == '\n') || (*buf == '\r') || (*buf == '\t')))
        ++buf;
    return *buf == '\0' ? -1 : 0;
}

/// return 0 if the first non-pad character is #
inline
int is_comment(char *buf)
{
    skip_pad(buf);
    if (buf[0] == '#')
        return 1;
    return 0;
}

/// A traits class for scanf conversion codes.
template <typename num_t>
struct TECA_EXPORT scanf_tt {};

#define DECLARE_SCANF_TT(_CPP_T, _FMT_STR)                              \
template<>                                                              \
/** A traits class for scanf conversion codes, specialized fo _CPP_T */ \
struct scanf_tt<_CPP_T>                                                 \
{                                                                       \
    static                                                              \
    const char *format() { return _FMT_STR; }                           \
};
DECLARE_SCANF_TT(float," %g")
DECLARE_SCANF_TT(double," %lg")
DECLARE_SCANF_TT(char," %hhi")
DECLARE_SCANF_TT(short, " %hi")
DECLARE_SCANF_TT(int, " %i")
DECLARE_SCANF_TT(long, " %li")
DECLARE_SCANF_TT(long long, "%lli")
DECLARE_SCANF_TT(unsigned char," %hhu")
DECLARE_SCANF_TT(unsigned short, " %hu")
DECLARE_SCANF_TT(unsigned int, " %u")
DECLARE_SCANF_TT(unsigned long, " %lu")
DECLARE_SCANF_TT(unsigned long long, "%llu")
DECLARE_SCANF_TT(std::string, " \"%128s")

/// A traits class for conversion from text to numbers
template <typename T>
struct TECA_EXPORT string_tt {};

#define DECLARE_STR_CONVERSION_I(_CPP_T, _FUNC)                                     \
/** A traits class for conversion from text to numbers, specialized for _CPP_T */   \
template <>                                                                         \
struct string_tt<_CPP_T>                                                            \
{                                                                                   \
    static const char *type_name() { return # _CPP_T; }                             \
                                                                                    \
    static int convert(const char *str, _CPP_T &val)                                \
    {                                                                               \
        errno = 0;                                                                  \
        char *endp = nullptr;                                                       \
        _CPP_T tmp = _FUNC(str, &endp, 0);                                          \
        if (errno != 0)                                                             \
        {                                                                           \
            TECA_ERROR("Failed to convert string \""                                \
                << str << "\" to a nunber." << strerror(errno))                     \
            return  -1;                                                             \
        }                                                                           \
        else if (endp == str)                                                       \
        {                                                                           \
            TECA_ERROR("Failed to convert string \""                                \
                << str << "\" to a nunber. Invalid string.")                        \
            return  -1;                                                             \
        }                                                                           \
        val = tmp;                                                                  \
        return 0;                                                                   \
    }                                                                               \
};

#define DECLARE_STR_CONVERSION_F(_CPP_T, _FUNC)                                     \
/** A traits class for conversion from text to numbers, specialized for _CPP_T */   \
template <>                                                                         \
struct string_tt<_CPP_T>                                                            \
{                                                                                   \
    static const char *type_name() { return # _CPP_T; }                             \
                                                                                    \
    static int convert(const char *str, _CPP_T &val)                                \
    {                                                                               \
        errno = 0;                                                                  \
        char *endp = nullptr;                                                       \
        _CPP_T tmp = _FUNC(str, &endp);                                             \
        if (errno != 0)                                                             \
        {                                                                           \
            TECA_ERROR("Failed to convert string \""                                \
                << str << "\" to a nunber." << strerror(errno))                     \
            return  -1;                                                             \
        }                                                                           \
        else if (endp == str)                                                       \
        {                                                                           \
            TECA_ERROR("Failed to convert string \""                                \
                << str << "\" to a nunber. Invalid string.")                        \
            return  -1;                                                             \
        }                                                                           \
        val = tmp;                                                                  \
        return 0;                                                                   \
    }                                                                               \
};

DECLARE_STR_CONVERSION_F(float, strtof)
DECLARE_STR_CONVERSION_F(double, strtod)
DECLARE_STR_CONVERSION_I(char, strtol)
DECLARE_STR_CONVERSION_I(short, strtol)
DECLARE_STR_CONVERSION_I(int, strtol)
DECLARE_STR_CONVERSION_I(long, strtoll)
DECLARE_STR_CONVERSION_I(long long, strtoll)

/// A traits class for conversion from text to numbers, specialized for bool
template <>
struct string_tt<bool>
{
    static const char *type_name() { return "bool"; }

    static int convert(const char *str, bool &val)
    {
        char buf[17];
        buf[16] = '\0';
        size_t n = strlen(str);
        n = n < 17 ? n : 16;
        for (size_t i = 0; i < n && i < 16; ++i)
            buf[i] = tolower(str[i]);
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

        TECA_ERROR("Failed to convert string \"" << str << "\" to a bool")
        return -1;
    }
};

/// A traits class for conversion from text to numbers, specialized for std::string
template <>
struct string_tt<std::string>
{
    static const char *type_name() { return "std::string"; }

    static int convert(const char *str, std::string &val)
    {
        val = str;
        return 0;
    }
};

/** A traits class for conversion from text to numbers, specialized for char*
 * watch out for memory leak, val needs to be free'd
 */
template <>
struct string_tt<char*>
{
    static const char *type_name() { return "char*"; }

    static int convert(const char *str, char *&val)
    {
        val = strdup(str);
        return 0;
    }
};

/** Extract the value in a "name = value" pair.
 * an error occurs if splitting the input on '=' doesn't produce 2 tokens
 * or if the conversion to val_t fails. returns 0 if successful.
 */
template <typename val_t>
TECA_EXPORT
int extract_value(char *l, val_t &val)
{
    std::vector<char*> tmp;
    if (tokenize(l, '=', tmp) || (tmp.size() != 2))
    {
        TECA_ERROR("Invalid name specifier in \"" << l << "\"")
        return -1;
    }

    char *r = tmp[1];
    if (skip_pad(r) || string_tt<val_t>::convert(r, val))
    {
        TECA_ERROR("Invalid " << string_tt<val_t>::type_name()
            << " value \"" << r << "\" in \"" << l << "\"")
        return -1;
    }

    return 0;
}

/** Given a collection of strings, where some of the strings end with a common
 * substring, the post-fix, this function visits each string in the collection
 * and removes the post-fix from each string that it is found in.
 */
TECA_EXPORT
void remove_postfix(std::set<std::string> &names, std::string postfix);

/// When passed the string "" return empty string otherwise return the passed string
TECA_EXPORT
inline std::string emptystr(const std::string &in)
{
    return (in == "\"\"" ? std::string() : in);
}

}

#endif
