#ifndef teca_file_util_h
#define teca_file_util_h

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

class teca_binary_stream;

#ifndef WIN32
  #define PATH_SEP "/"
#else
  #define PATH_SEP "\\"
#endif

namespace teca_file_util
{
// read the file into a stream. read will fail if header
// is not found. return zero upon success. The verbose flag
// indicates whether or not an error is reported if opening
// the file fails. All other errors are always reported.
int read_stream(const char *file_name, const char *header,
    teca_binary_stream &stream, bool verbose=true);

// write the stream to the file. the header is prepended to the
// file. return zero upon success. The verbose flag indicates
// whether or not an error is reported if creating the file
// fails. All other errors are reported.
int write_stream(const char *file_name, const char *header,
    const teca_binary_stream &stream, bool verbose=true);

// replace %t% with the given value
void replace_timestep(std::string &file_name, unsigned long time_step);

// replace %e% with the given string
void replace_extension(std::string &file_name, const std::string &ext);

// replace %s% with the given string
void replace_identifier(std::string &file_name, const std::string &id);

// return string converted to lower case
void to_lower(std::string &in);

// return 0 if the file does not exist
int file_exists(const char *path);

// return 0 if the file/directory is not writeable
int file_writable(const char *path);

// Returns the path not including the file name and not
// including the final PATH_SEP. If PATH_SEP isn't found
// then ".PATH_SEP" is returned.
std::string path(const std::string &filename);

// Returns the file name not including the extension (ie what ever is after
// the last ".". If there is no "." then the filename is returned unmodified.
std::string base_filename(const std::string &filename);

// Returns the file name from the given path. If PATH_SEP isn't found
// then the filename is returned unmodified.
std::string filename(const std::string &filename);

// Returns the extension from the given filename.
std::string extension(const std::string &filename);

// rad the line of the ascii file into a vector
size_t load_lines(const char *filename, std::vector<std::string> &lines);

// read the file into a string
size_t load_text(const std::string &filename, std::string &text);

// write the string to the named file
int write_text(std::string &filename, std::string &text);

int search_and_replace(
    const std::string &search_for,
    const std::string &replace_with,
    std::string &in_text);

int locate_files(
    const std::string &path,
    const std::string &re,
    std::vector<std::string> &file_list);

//*****************************************************************************
template<typename T>
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

/**
*/
// ****************************************************************************
template<typename T>
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

/**
Parse a string for a "key", starting at offset "at" then
advance past the key and attempt to convert what follows
in to a value of type "T". If the key isn't found, then
npos is returned otherwise the position imediately following
the key is returned.
*/
// ****************************************************************************
template <typename T>
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

};

#endif
