#include "teca_config.h"
#include "teca_common.h"
#include "teca_binary_stream.h"

#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstring>
#include <errno.h>

#if defined(TECA_HAS_REGEX)
#include <regex>
#endif

#ifndef WIN32
  #include <fcntl.h>
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
#if !defined(TECA_HAS_REGEX)
    (void)code;
    return "c++11 regex support is disabled in this build of TECA";
#else
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
#endif
}

// **************************************************************************
#if defined(TECA_HAS_REGEX)
std::regex filter_regex;
#endif
int set_filter_regex(const std::string &re)
{
#if !defined(TECA_HAS_REGEX)
    (void)re;
    TECA_ERROR(
        << "Failed to compile regular expression" << std::endl
        << re << std::endl
        << "This compiler does not have regex support.")
    return -1;
#else
    try
    {
       teca_file_util::filter_regex = std::regex(re, std::regex::grep);
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
#endif
}

// **************************************************************************
int scandir_filter(const struct dirent *de)
{
#if !defined(TECA_HAS_REGEX)
    (void)de;
    return 0;
#else
    std::cmatch matches;
    if (std::regex_search(de->d_name, matches, teca_file_util::filter_regex))
        return 1;
    return 0;
#endif
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
void replace_timestep(std::string &file_name, unsigned long time_step)
{
    size_t t_pos = file_name.find("%t%");
    if (t_pos != std::string::npos)
    {
        std::ostringstream oss;
        oss << time_step;

        file_name.replace(t_pos, 3, oss.str());
    }
}

// **************************************************************************
void replace_extension(std::string &file_name, const std::string &ext)
{
    size_t ext_pos = file_name.find("%e%");
    if (ext_pos != std::string::npos)
        file_name.replace(ext_pos, 3, ext);
}

// **************************************************************************
void replace_identifier(std::string &file_name, const std::string &id)
{
    size_t ext_pos = file_name.find("%s%");
    if (ext_pos != std::string::npos)
        file_name.replace(ext_pos, 3, id);
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
        return 1;
#else
    (void)path;
#endif
    return 0;
}

// ***************************************************************************
int file_writable(const char *path)
{
#ifndef WIN32
    if (!access(path, W_OK))
        return 1;
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

// ***************************************************************************
std::string path(const std::string &filename)
{
    size_t p;
    p = filename.find_last_of(PATH_SEP);
    if (p == std::string::npos)
        return "." PATH_SEP;

    return filename.substr(0,p);
}

// ***************************************************************************
std::string base_filename(const std::string &filename)
{
    size_t p;
    p = filename.rfind(".");
    if (p == std::string::npos)
        return filename;

    return filename.substr(0, p);
}

// ***************************************************************************
std::string filename(const std::string &filenm)
{
    size_t p;
    p = filenm.find_last_of(PATH_SEP);
    if (p == std::string::npos)
        return filenm;

    return filenm.substr(p+1,std::string::npos);
}

// ***************************************************************************
std::string extension(const std::string &filename)
{
    size_t p;
    p = filename.rfind(".");
    if (p == std::string::npos)
        return "";

    return filename.substr(p+1);
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
        TECA_ERROR("File " << filename << " could not be opened.")
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
        TECA_ERROR("File " << filename << " could not be opened.")
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

// **************************************************************************
int read_stream(const char *file_name, const char *header,
    teca_binary_stream &stream, bool verbose)
{
    // open the file
    FILE* fd = fopen(file_name, "rb");
    if (fd == NULL)
    {
        if (verbose)
        {
            const char *estr = strerror(errno);
            TECA_ERROR("Failed to open " << file_name << ". " << estr)
        }
        return -1;
    }

    // get its length, we'll read it in one go and need to create
    // a bufffer for it's contents
    unsigned long start = ftell(fd);
    fseek(fd, 0, SEEK_END);
    unsigned long end = ftell(fd);
    fseek(fd, 0, SEEK_SET);

    // look at the header to check if this is really ours
    unsigned long header_len = strlen(header);

    char *file_header = static_cast<char*>(malloc(header_len+1));
    file_header[header_len] = '\0';

    if (fread(file_header, 1, header_len, fd) != header_len)
    {
        const char *estr = (ferror(fd) ? strerror(errno) : "");
        fclose(fd);
        free(file_header);
        TECA_ERROR("Failed to read header from \""
            << file_name << "\". " << estr)
        return -1;
    }

    if (strncmp(file_header, header, header_len))
    {
        fclose(fd);
        free(file_header);
        TECA_ERROR("Header missmatch in \""
             << file_name << "\". Expected \"" << header
             << "\" found \"" << file_header << "\"")
        return -1;
    }

    free(file_header);

    // create the buffer for the file contents
    unsigned long nbytes = end - start - header_len;
    stream.resize(nbytes);

    // read the stream
    unsigned long bytes_read =
        fread(stream.get_data(), sizeof(unsigned char), nbytes, fd);

    if (bytes_read != nbytes)
    {
        const char *estr = (ferror(fd) ? strerror(errno) : "");
        fclose(fd);
        TECA_ERROR("Failed to read \"" << file_name << "\". Read only "
            << bytes_read << " of the requested " << nbytes << ". " << estr)
        return -1;
    }

    // update the stream to reflect it's new contents
    stream.set_read_pos(0);
    stream.set_write_pos(nbytes);

    // close the file
    if (fclose(fd))
    {
        const char *estr = strerror(errno);
        TECA_ERROR("Failed to close \"" << file_name << "\". " << estr)
        return -1;
    }


    return 0;
}

// **************************************************************************
int write_stream(const char *file_name, const char *header,
    const teca_binary_stream &stream, bool verbose)
{
    // open up a file
    int fd = creat(file_name, S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH);
    if (fd == -1)
    {
        if (verbose)
        {
            const char *estr = strerror(errno);
            TECA_ERROR("Failed to create \"" << file_name << "\". " << estr)
        }
        return -1;
    }

    // this will let the reader verify that we have a teca binary table
    long header_len = strlen(header);
    if (write(fd, header, header_len) != header_len)
    {
        const char *estr = strerror(errno);
        TECA_ERROR("Failed to write header to \""
            << file_name << "\". " << estr)
        return -1;
    }

    // now write the table
    ssize_t n_wrote = 0;
    ssize_t n_to_write = stream.size();
    while (n_to_write > 0)
    {
        ssize_t n = write(fd, stream.get_data() + n_wrote, n_to_write);
        if (n == -1)
        {
            const char *estr = strerror(errno);
            TECA_ERROR("Failed to write \"" << file_name << "\". " << estr)
            return -1;
        }
        n_wrote += n;
        n_to_write -= n;
    }

    // and close the file out
    if (close(fd))
    {
        const char *estr = strerror(errno);
        TECA_ERROR("Failed to close \"" << file_name << "\". " << estr)
        return -1;
    }

    return 0;
}

};
