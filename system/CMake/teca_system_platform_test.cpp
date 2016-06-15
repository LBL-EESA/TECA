#ifdef TEST_TECA_CXX_HAS_BACKTRACE
#if defined(__PATHSCALE__) || defined(__PATHCC__) \
  || (defined(__LSB_VERSION__) && (__LSB_VERSION__ < 41))
backtrace doesnt work with this compiler or os
#endif
#if (defined(__GNUC__) || defined(__PGI)) && !defined(_GNU_SOURCE)
# define _GNU_SOURCE
#endif
#include <execinfo.h>
int main()
{
  void *stackSymbols[256];
  backtrace(stackSymbols,256);
  backtrace_symbols(&stackSymbols[0],1);
  return 0;
}
#endif

#ifdef TEST_TECA_CXX_HAS_DLADDR
#if (defined(__GNUC__) || defined(__PGI)) && !defined(_GNU_SOURCE)
# define _GNU_SOURCE
#endif
#include <dlfcn.h>
int main()
{
  Dl_info info;
  int ierr=dladdr((void*)main,&info);
  return 0;
}
#endif

#ifdef TEST_TECA_CXX_HAS_CXXABI
#if (defined(__GNUC__) || defined(__PGI)) && !defined(_GNU_SOURCE)
# define _GNU_SOURCE
#endif
#include <cxxabi.h>
int main()
{
  int status = 0;
  size_t bufferLen = 512;
  char buffer[512] = {'\0'};
  const char *function="_ZN5kwsys17SystemInformation15GetProgramStackEii";
  char *demangledFunction =
    abi::__cxa_demangle(function, buffer, &bufferLen, &status);
  return status;
}
#endif
