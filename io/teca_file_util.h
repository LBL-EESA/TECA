#ifndef teca_file_util_h
#define teca_file_util_h

/// @file

#include "teca_config.h"

#include <vector>
#include <deque>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <sys/stat.h>

class teca_binary_stream;

#ifndef WIN32
  #define PATH_SEP "/"
#else
  #define PATH_SEP "\\"
#endif

/// Codes dealing with low level file system API's
namespace teca_file_util
{
/** read the file into a stream. if header is not null the call will fail if the
 * given string is not found. return zero upon success. The verbose flag
 * indicates whether or not an error is reported if opening the file fails. All
 * other errors are always reported.
 */
TECA_EXPORT
int read_stream(const char *file_name, const char *header,
    teca_binary_stream &stream, bool verbose=true);

/** write the stream to the file. the passed in flags control file access, a
 * reasonable value is S_IRUSR|S_IWUSR|S_IRGRP|S_IROTH. if header is not null
 * the given string is prepended to the file. return zero upon success. The
 * verbose flag indicates whether or not an error is reported if creating the
 * file fails. All other errors are reported.
 */
TECA_EXPORT
int write_stream(const char *file_name, int flags, const char *header,
    const teca_binary_stream &stream, bool verbose=true);

/// replace %t% with the given value
TECA_EXPORT
void replace_timestep(std::string &file_name, unsigned long time_step, int width = 6);

/// replace %t% with the time t in calendar with units in the strftime format
TECA_EXPORT
int replace_time(std::string &file_name, double t,
    const std::string &calendar, const std::string &units,
    const std::string &format);

/// replace %e% with the given string
TECA_EXPORT
void replace_extension(std::string &file_name, const std::string &ext);

/// replace %s% with the given string
TECA_EXPORT
void replace_identifier(std::string &file_name, const std::string &id);

/// return string converted to lower case
TECA_EXPORT
void to_lower(std::string &in);

/// return 0 if the file does not exist
TECA_EXPORT
int file_exists(const char *path);

/// return 0 if the file/directory is not writeable
TECA_EXPORT
int file_writable(const char *path);

/** Returns the path not including the file name and not including the final
 * PATH_SEP. If PATH_SEP isn't found then ".PATH_SEP" is returned.
 */
TECA_EXPORT
std::string path(const std::string &filename);

/** Returns the file name not including the extension (ie what ever is after
 * the last ".". If there is no "." then the filename is returned unmodified.
 */
TECA_EXPORT
std::string base_filename(const std::string &filename);

/** Returns the file name from the given path. If PATH_SEP isn't found
 * then the filename is returned unmodified.
 */
TECA_EXPORT
std::string filename(const std::string &filename);

/// Returns the extension from the given filename.
TECA_EXPORT
std::string extension(const std::string &filename);

/// read the lines of the ascii file into a vector
TECA_EXPORT
size_t load_lines(const char *filename, std::vector<std::string> &lines);

/// read the file into a string
TECA_EXPORT
size_t load_text(const std::string &filename, std::string &text);

/// write the string to the named file
TECA_EXPORT
int write_text(std::string &filename, std::string &text);

/// Search and replace with in a string of text.
TECA_EXPORT
int search_and_replace(
    const std::string &search_for,
    const std::string &replace_with,
    std::string &in_text);

/// Locate files in path that match a regular expression.
TECA_EXPORT
int locate_files(
    const std::string &path,
    const std::string &re,
    std::vector<std::string> &file_list);

/// Load a binary file into memory
template<typename T>
TECA_EXPORT
size_t load_bin(const char *filename, size_t dlen, T *buffer)
{
  std::ifstream file(filename,std::ios::binary);
  if (!file.is_open())
    {
    std::cerr << "ERROR: File " << filename << " could not be opened." << std::endl;
    return 0;
    }

  // determine file size
  file.seekg(0,std::ios::end);
  size_t flen=file.tellg();
  file.seekg(0,std::ios::beg);

  // check if file size matches expected read size.
  if (dlen*sizeof(T)!=flen)
    {
    std::cerr
      << "ERROR: Expected " << dlen << " bytes but found "
      << flen << " bytes in \"" << filename << "\".";
    return 0;
    }

  // read
  file.read((char*)buffer,flen);
  file.close();

  // return the data, it's up to the caller to free.
  return dlen;
}

/// extract a name-value pair from the given set of lines.
template<typename T>
TECA_EXPORT
int name_value(std::vector<std::string> &lines, std::string name, T &value)
{
  size_t n_lines=lines.size();
  for (size_t i=0; i<n_lines; ++i)
    {
    std::string tok;
    std::istringstream is(lines[i]);
    is >> tok;
    if (tok==name)
      {
      is >> value;
      return 1;
      }
    }
  return 0;
}

/** Parse a string for a "key", starting at offset "at" then advance past the
 * key and attempt to convert what follows in to a value of type "T". If the
 * key isn't found, then npos is returned otherwise the position imediately
 * following the key is returned.
*/
template <typename T>
TECA_EXPORT
size_t parse_value(std::string &in,size_t at, std::string key, T &value)
{
  size_t p=in.find(key,at);
  if (p!=std::string::npos)
    {
    size_t n=key.size();

    // check to make sure match is the whole word
    if ((p!=0) && isalpha(in[p-1]) && isalpha(in[p+n]))
      {
      return std::string::npos;
      }
    // convert value
    const int max_value_len=64;
    p+=n;
    std::istringstream valss(in.substr(p,max_value_len));

    valss >> value;
    }
  return p;
}

/** a stack of lines. lines can be popped as they are processed and the current
 * line number is recorded.
 */
struct TECA_EXPORT line_buffer
{
    line_buffer() : m_buffer(nullptr), m_line_number(0) {}
    ~line_buffer() { free(m_buffer); }

    // read the contents of the file and intitalize the
    // stack of lines.
    int initialize(const char *file_name);

    // check if the stack is not empty
    operator bool ()
    {
        return !m_lines.empty();
    }

    // get the line at the top of the stack
    char *current()
    {
        return m_lines.front();
    }

    // remove the line at the top of the stack
    void pop()
    {
        m_lines.pop_front();
        ++m_line_number;
    }

    // get the current line number
    size_t line_number()
    {
        return m_line_number;
    }

    char *m_buffer;
    std::deque<char*> m_lines;
    size_t m_line_number;
};

};

#endif
