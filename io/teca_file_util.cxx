#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <regex>
#include <cstring>
#include <errno.h>
#include "teca_common.h"

#ifndef WIN32
  #include <dirent.h>
  #define PATH_SEP "/"
  #include <sys/types.h>
  #include <sys/stat.h>
  #include <unistd.h>
#else
  #include "win_windirent.h"
  #define opendir win_opendir
  #define readdir win_readdir
  #define closedir win_closedir
  #define DIR win_DIR
  #define dirent win_dirent
  #define PATH_SEP "\\"
#endif

namespace teca_file_util {

// **************************************************************************
const char *regex_strerr(int code)
{
    switch (code)
    {
    case std::regex_constants::error_collate:
        return "The expression contained an invalid collating element name.";
    case std::regex_constants::error_ctype:
        return "The expression contained an invalid character class name.";
    case std::regex_constants::error_escape:
        return "The expression contained an invalid escaped character, or a"
               "trailing escape.";
    case std::regex_constants::error_backref:
        return "The expression contained an invalid back reference.";
    case std::regex_constants::error_brack:
        return "The expression contained mismatched brackets ([ and ]).";
    case std::regex_constants::error_paren:
        return "The expression contained mismatched parentheses (( and )).";
    case std::regex_constants::error_brace:
        return "The expression contained mismatched braces ({ and }).";
    case std::regex_constants::error_badbrace:
        return "The expression contained an invalid range between braces ({ and }).";
    case std::regex_constants::error_range:
        return "The expression contained an invalid character range.";
    case std::regex_constants::error_space:
        return "There was insufficient memory to convert the expression into a"
               " finite state machine.";
    case std::regex_constants::error_badrepeat:
        return "The expression contained a repeat specifier (one of *?+{) that"
               " was not preceded by a valid regular expression.";
    case std::regex_constants::error_complexity:
        return "The complexity of an attempted match against a regular"
               " expression exceeded a pre-set level.";
    case std::regex_constants::error_stack:
        return "There was insufficient memory to determine whether the regular"
               " expression could match the specified character sequence.";
    }
    return "unkown regex error";
}

// **************************************************************************
std::regex filter_regex;
int set_filter_regex(const std::string &re)
{
    try
    {
       teca_file_util::filter_regex = std::regex(re);
    }
    catch (std::regex_error &e)
    {
        TECA_ERROR(
            << "Failed to compile regular expression" << std::endl
            << re << std::endl
            << regex_strerr(e.code()))
        return -1;
    }
    return 0;
}

// **************************************************************************
int scandir_filter(const struct dirent *de)
{
    std::cmatch matches;
    if (std::regex_search(de->d_name, matches, teca_file_util::filter_regex))
        return 1;
    return 0;
}

// **************************************************************************
int locate_files(
    const std::string &path,
    const std::string &re,
    std::vector<std::string> &file_list)
{
    if (teca_file_util::set_filter_regex(re))
        return -1;

    struct dirent **entries;

    int n_files = scandir(
            path.c_str(),
            &entries,
            teca_file_util::scandir_filter,
#ifdef __APPLE__
            alphasort
#else
            versionsort
#endif
            );

    if (n_files < 0)
    {
        int e = errno;
        TECA_ERROR("Failed to scan for files" << std::endl << strerror(e))
        return -2;
    }
    else
    if (n_files < 1)
    {
        TECA_ERROR("Found no files matching regular expression" << std::endl << re)
        return -3;
    }

    for (int i = 0; i < n_files; ++i)
    {
        file_list.push_back(entries[i]->d_name);
        free(entries[i]);
        entries[i] = nullptr;
    }
    free(entries);

    return 0;
}

// **************************************************************************
void to_lower(std::string &in)
{
    size_t n = in.size();
    for (size_t i = 0; i < n; ++i)
        in[i] = (char)tolower(in[i]);
}

// ***************************************************************************
int file_exists(const char *path)
{
    #ifndef WIN32
    struct stat s;
    int i_err = stat(path, &s);
    if (i_err == 0)
    {
        return 1;
    }
    #else
    (void)path;
    #endif
    return 0;
}

// ***************************************************************************
int present(const char *path, const char *filename, const char *ext)
{
  std::ostringstream fn;
  fn << path << PATH_SEP << filename << "." << ext;
  FILE *fp = fopen(fn.str().c_str(),"r");
  if (fp == 0)
    {
    // file is not present.
    return 0;
    }
  // file is present.
  fclose(fp);
  return 1;
}


// Returns the path not including the file name and not
// including the final PATH_SEP. If PATH_SEP isn't found
// then ".PATH_SEP" is returned.
// ***************************************************************************
std::string strip_filename_from_path(const std::string &filename)
{
    size_t p;
    p = filename.find_last_of(PATH_SEP);
    if (p == std::string::npos)
        return "." PATH_SEP;

    return filename.substr(0,p);
}

// Returns the file name not including the extension (ie what ever is after
// the last ".". If there is no "." then the filename is retnurned unmodified.
// ***************************************************************************
std::string strip_extension_from_filename(const std::string &filename)
{
    size_t p;
    p = filename.rfind(".");
    if (p == std::string::npos)
        return filename;

    return filename.substr(0, p);
}

// Returns the file name from the given path. If PATH_SEP isn't found
// then the filename is returned unmodified.
// ***************************************************************************
std::string strip_path_from_filename(const std::string &filename)
{
    size_t p;
    p = filename.find_last_of(PATH_SEP);
    if (p == std::string::npos)
        return filename;

    return filename.substr(p+1,std::string::npos);
}


// **************************************************************************
size_t load_lines(const char *filename, std::vector<std::string> &lines)
{
    // Load each line in the given file into a the vector.
    size_t n_read = 0;
    const int buf_size = 1024;
    char buf[buf_size] = {'\0'};
    std::ifstream file(filename);
    if (!file.is_open())
    {
        std::cerr << "ERROR: File " << filename << " could not be opened." << std::endl;
        return 0;
    }
    while(file.good())
    {
        file.getline(buf,buf_size);
        if (file.gcount() > 1)
        {
            lines.push_back(buf);
            ++n_read;
        }
    }
    file.close();
    return n_read;
}

// **************************************************************************
size_t load_text(const std::string &filename, std::string &text)
{
    std::ifstream file(filename.c_str());
    if (!file.is_open())
    {
        std::cerr << "ERROR: File " << filename << " could not be opened." << std::endl;
        return 0;
    }
    // Determine the length of the file ...
    file.seekg (0, std::ios::end);
    size_t n_bytes = (size_t)file.tellg();
    file.seekg (0, std::ios::beg);
    // and allocate a buffer to hold its contents.
    char *buf = new char [n_bytes];
    memset(buf, 0, n_bytes);
    // Read the file and convert to a string.
    file.read (buf, n_bytes);
    file.close();
    text = buf;
    return n_bytes;
}

// **************************************************************************
int write_text(std::string &filename, std::string &text)
{
    std::ofstream file(filename.c_str());
    if (!file.is_open())
    {
        std::cerr << "ERROR: File " << filename << " could not be opened." << std::endl;
        return 0;
    }
    file << text << std::endl;
    file.close();
    return 1;
}

// **************************************************************************
int search_and_replace(
        const std::string &search_for,
        const std::string &replace_with,
        std::string &in_text)
{
    int n_found = 0;
    const size_t n = search_for.size();
    size_t at=std::string::npos;
    while ((at = in_text.find(search_for)) != std::string::npos)
    {
        in_text.replace(at,n,replace_with);
        ++n_found;
    }
    return n_found;
}

};
