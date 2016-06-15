#include "teca_system_interface.h"
#include "teca_common.h"

#if defined(_WIN32)
# define NOMINMAX // use our min, max
# if !defined(_WIN32_WINNT) && !(defined(_MSC_VER) && _MSC_VER < 1300)
#  define _WIN32_WINNT 0x0501
# endif
# include <winsock.h> // WSADATA,  include before sys/types.h
#endif

#if (defined(__GNUC__) || defined(__PGI)) && !defined(_GNU_SOURCE)
# define _GNU_SOURCE
#endif

#include <string>
#include <vector>
#include <iosfwd>
#include <iostream>
#include <sstream>
#include <fstream>

#if defined(_WIN32)
# include <windows.h>
# if defined(_MSC_VER) && _MSC_VER >= 1800
#  define TECA_WINDOWS_DEPRECATED_GetVersionEx
# endif
# include <errno.h>
# if defined(TECA_SYS_HAS_PSAPI)
#  include <psapi.h>
# endif
# if !defined(siginfo_t)
typedef int siginfo_t;
# endif
#else
# include <sys/types.h>
# include <sys/time.h>
# include <sys/utsname.h> // int uname(struct utsname *buf);
# include <sys/resource.h> // getrlimit
# include <unistd.h>
# include <signal.h>
# include <fcntl.h>
# include <errno.h> // extern int errno;
#endif

#ifdef __FreeBSD__
# include <sys/sysctl.h>
# include <fenv.h>
# include <sys/socket.h>
# include <netdb.h>
# include <netinet/in.h>
# if defined(TECA_SYS_HAS_IFADDRS_H)
#  include <ifaddrs.h>
#  define TECA_IMPLEMENT_FQDN
# endif
#endif

#if defined(__OpenBSD__) || defined(__NetBSD__)
# include <sys/param.h>
# include <sys/sysctl.h>
#endif

#if defined(TECA_SYS_HAS_MACHINE_CPU_H)
# include <machine/cpu.h>
#endif

#if defined(__DragonFly)
# include <sys/sysctl.h>
#endif

#ifdef __APPLE__
# include <sys/sysctl.h>
# include <mach/vm_statistics.h>
# include <mach/host_info.h>
# include <mach/mach.h>
# include <mach/mach_types.h>
# include <fenv.h>
# include <sys/socket.h>
# include <netdb.h>
# include <netinet/in.h>
# if defined(TECA_SYS_HAS_IFADDRS_H)
#  include <ifaddrs.h>
#  define TECA_IMPLEMENT_FQDN
# endif
# if !(__ENVIRONMENT_MAC_OS_X_VERSION_MIN_REQUIRED__-0 >= 1050)
#  undef TECA_HAS_BACKTRACE
# endif
#endif

#ifdef __linux
# include <fenv.h>
# include <sys/socket.h>
# include <netdb.h>
# include <netinet/in.h>
# if defined(TECA_SYS_HAS_IFADDRS_H)
#  include <ifaddrs.h>
#  if !defined(__LSB_VERSION__) /* LSB has no getifaddrs */
#   define TECA_IMPLEMENT_FQDN
#  endif
# endif
# if defined(TECA_CXX_HAS_RLIMIT64)
typedef struct rlimit64 resource_limit_type;
#  define get_resource_limit getrlimit64
# else
typedef struct rlimit resource_limit_type;
#  define get_resource_limit getrlimit
# endif
#endif

#ifdef __HAIKU__
# include <OS.h>
#endif

#if defined(TECA_HAS_BACKTRACE)
# include <execinfo.h>
# if defined(TECA_HAS_CPP_DEMANGLE)
#  include <cxxabi.h>
# endif
# if defined(TECA_HAS_SYMBOL_LOOKUP)
#  include <dlfcn.h>
# endif
#else
# undef TECA_HAS_CPP_DEMANGLE
# undef TECA_HAS_SYMBOL_LOOKUP
#endif

#include <memory.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h> // int isdigit(int c);

extern "C" { typedef void (*sig_action_t)(int,siginfo_t*,void*); }

namespace {

// ****************************************************************************
#if !defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)
void stacktrace_signal_handler(int sig_no,
     siginfo_t *sig_info, void * /*sig_context*/)
{
#if defined(__linux) || defined(__APPLE__)
    std::ostringstream oss;
    oss << std::endl
       << "=========================================================" << std::endl
       << "process id " << getpid() << " ";
    switch (sig_no)
    {
        case SIGINT:
            oss << "caught SIGINT";
            break;

        case SIGTERM:
            oss << "caught SIGTERM";
            break;

        case SIGABRT:
            oss << "caught SIGABRT";
            break;

        case SIGFPE:
            oss << "caught SIGFPE at "
                << (sig_info->si_addr == 0?"0x":"")
                << sig_info->si_addr <<  " ";
            switch (sig_info->si_code)
            {
#if defined(FPE_INTDIV)
                case FPE_INTDIV:
                    oss << "integer division by zero";
                    break;
#endif
#if defined(FPE_INTOVF)
                case FPE_INTOVF:
                    oss << "integer overflow";
                    break;
#endif
                case FPE_FLTDIV:
                    oss << "floating point divide by zero";
                    break;

                case FPE_FLTOVF:
                    oss << "floating point overflow";
                    break;

                case FPE_FLTUND:
                    oss << "floating point underflow";
                    break;

                case FPE_FLTRES:
                    oss << "floating point inexact result";
                    break;

                case FPE_FLTINV:
                    oss << "floating point invalid operation";
                    break;
#if defined(FPE_FLTSUB)
                case FPE_FLTSUB:
                    oss << "floating point subscript out of range";
                    break;
#endif
                default:
                    oss << "code " << sig_info->si_code;
                    break;
            }
            break;

        case SIGSEGV:
            oss << "caught SIGSEGV at "
                << (sig_info->si_addr == 0 ? "0x" : "")
                << sig_info->si_addr <<  " ";
            switch (sig_info->si_code)
            {
                case SEGV_MAPERR:
                    oss << "address not mapped to object";
                    break;

                case SEGV_ACCERR:
                    oss << "invalid permission for mapped object";
                    break;

                default:
                    oss << "code " << sig_info->si_code;
                    break;
            }
            break;

        case SIGBUS:
            oss << "caught SIGBUS at "
                << (sig_info->si_addr == 0?"0x":"")
                << sig_info->si_addr <<  " ";
            switch (sig_info->si_code)
            {
                case BUS_ADRALN:
                  oss << "invalid address alignment";
                  break;
#if defined(BUS_ADRERR)
                case BUS_ADRERR:
                  oss << "nonexistent physical address";
                  break;
#endif
# if defined(BUS_OBJERR)
                case BUS_OBJERR:
                  oss << "object-specific hardware error";
                  break;
#endif
#if defined(BUS_MCEERR_AR)
                case BUS_MCEERR_AR:
                  oss << "hardware memory error consumed on a machine check; action required.";
                  break;
#endif
#if defined(BUS_MCEERR_AO)
                case BUS_MCEERR_AO:
                  oss << "hardware memory error detected in process but not consumed; action optional.";
                  break;
#endif
                default:
                  oss << "code " << sig_info->si_code;
                  break;
            }
            break;

        case SIGILL:
            oss << "caught SIGILL at "
                << (sig_info->si_addr == 0 ? "0x" : "")
                << sig_info->si_addr <<  " ";
            switch (sig_info->si_code)
            {
                case ILL_ILLOPC:
                    oss << "illegal opcode";
                    break;
#if defined(ILL_ILLOPN)
                case ILL_ILLOPN:
                    oss << "illegal operand";
                    break;
#endif
#if defined(ILL_ILLADR)
                case ILL_ILLADR:
                    oss << "illegal addressing mode.";
                    break;
#endif
                case ILL_ILLTRP:
                    oss << "illegal trap";
                    break;

                case ILL_PRVOPC:
                    oss << "privileged opcode";
                    break;

#if defined(ILL_PRVREG)
                case ILL_PRVREG:
                    oss << "privileged register";
                    break;
#endif
#if defined(ILL_COPROC)
                case ILL_COPROC:
                    oss << "co-processor error";
                    break;
#endif
#if defined(ILL_BADSTK)
                case ILL_BADSTK:
                    oss << "internal stack error";
                    break;
#endif
                default:
                    oss << "code " << sig_info->si_code;
                    break;
            }
            break;

        default:
            oss << "caught " << sig_no << " code " << sig_info->si_code;
            break;
    }
    oss << std::endl << "program stack:" << std::endl
        << teca_system_interface::get_program_stack(2, 0)
        << "=========================================================" << std::endl;

    // report
    std::cerr << oss.str() << std::endl;

    // restore the previously registered handlers
    // and abort
    teca_system_interface::set_stack_trace_on_error(0);
    abort();
#else
    // avoid warning C4100
    (void)sig_no;
    (void)sig_info;
#endif
}
#endif

#if defined(TECA_HAS_BACKTRACE)
#define safes(_arg)((_arg)?(_arg):"???")

// description:
// A container for symbol properties. each instance
// must be initialized.
class symbol_properties
{
public:
    symbol_properties();

    // description:
    // the symbol_properties instance must be initialized by
    // passing a stack address.
    void initialize(void *address);

    // description:
    // get the symbol's stack address.
    void *get_address() const { return this->address; }

    // description:
    // if not set paths will be removed. eg,  from a binary
    // or source file.
    void set_report_path(int rp){ this->report_path = rp; }

    // description:
    // set/get the name of the binary file that the symbol
    // is found in.
    void set_binary(const char *binary)
    { this->binary = safes(binary); }

    std::string get_binary() const;

    // description:
    // set the name of the function that the symbol is found in.
    // if c++ demangling is supported it will be demangled.
    void set_function(const char *function)
    { this->function = this->demangle(function); }

    std::string get_function() const
    { return this->function; }

    // description:
    // set/get the name of the source file where the symbol
    // is defined.
    void set_source_file(const char *sourcefile)
    { this->source_file = safes(sourcefile); }

    std::string get_source_file() const
    { return this->get_file_name(this->source_file); }

    // description:
    // set/get the line number where the symbol is defined
    void set_line_number(long linenumber){ this->line_number = linenumber; }
    long get_line_number() const { return this->line_number; }

    // description:
    // set the address where the biinary image is mapped
    // into memory.
    void set_binary_base_address(void *address)
    { this->binary_base_address = address; }

private:
    void *get_real_address() const
    { return (void*)((char*)this->address-(char*)this->binary_base_address); }

    std::string get_file_name(const std::string &path) const;
    std::string demangle(const char *symbol) const;

private:
    std::string binary;
    void *binary_base_address;
    void *address;
    std::string source_file;
    std::string function;
    long line_number;
    int report_path;
};

// --------------------------------------------------------------------------
std::ostream &operator<<(std::ostream &os, const symbol_properties &sp)
{
#if defined(TECA_HAS_SYMBOL_LOOKUP)
    os << std::hex << sp.get_address() << " : "
        << sp.get_function()
        << " [(" << sp.get_binary() << ") "
        << sp.get_source_file() << ":"
        << std::dec << sp.get_line_number() << "]";
#elif defined(TECA_HAS_BACKTRACE)
      void *addr = sp.get_address();
      char **syminfo = backtrace_symbols(&addr, 1);
      os << safes(syminfo[0]);
      free(syminfo);
#else
      (void)os;
      (void)sp;
#endif
    return os;
}

// --------------------------------------------------------------------------
symbol_properties::symbol_properties()
{
    // not using an initializer list
    // to avoid some PGI compiler warnings
    this->set_binary("???");
    this->set_binary_base_address(NULL);
    this->address = NULL;
    this->set_source_file("???");
    this->set_function("???");
    this->set_line_number(-1);
    this->set_report_path(0);
    // avoid PGI compiler warnings
    this->get_real_address();
    this->get_function();
    this->get_source_file();
    this->get_line_number();
}

// --------------------------------------------------------------------------
std::string symbol_properties::get_file_name(const std::string &path) const
{
    std::string file(path);
    if (!this->report_path)
    {
        size_t at = file.rfind("/");
        if (at != std::string::npos)
        {
            file = file.substr(at+1, std::string::npos);
        }
    }
    return file;
}

// --------------------------------------------------------------------------
std::string symbol_properties::get_binary() const
{
// only linux has proc fs
#if defined(__linux)
    if (this->binary=="/proc/self/exe")
    {
        std::string binary;
        char buf[1024] = {'\0'};
        ssize_t ll = 0;
        if ((ll = readlink("/proc/self/exe", buf, 1024))>0)
        {
            buf[ll]='\0';
            binary = buf;
        }
        else
        {
            binary="/proc/self/exe";
        }
        return this->get_file_name(binary);
    }
#endif
    return this->get_file_name(this->binary);
}

// --------------------------------------------------------------------------
std::string symbol_properties::demangle(const char *symbol) const
{
    std::string result = safes(symbol);
#if defined(TECA_HAS_CPP_DEMANGLE)
    int status = 0;
    size_t buffer_len = 1024;
    char *buffer = (char*)malloc(1024);
    char *demangled_symbol =
        abi::__cxa_demangle(symbol,  buffer,  &buffer_len,  &status);
    if (!status)
    {
        result = demangled_symbol;
    }
    free(buffer);
#else
    (void)symbol;
#endif
    return result;
}

// --------------------------------------------------------------------------
void symbol_properties::initialize(void *address)
{
    this->address = address;
#if defined(TECA_HAS_SYMBOL_LOOKUP)
    // first fallback option can demangle c++ functions
    Dl_info info;
    int ierr = dladdr(this->address, &info);
    if (ierr && info.dli_sname && info.dli_saddr)
    {
        this->set_binary(info.dli_fname);
        this->set_function(info.dli_sname);
    }
#else
    // second fallback use builtin backtrace_symbols
    // to decode the bactrace.
#endif
}
#endif
}

namespace teca_system_interface
{
// **************************************************************************
std::string get_program_stack(int first_frame, int whole_path)
{
    std::string program_stack = ""
#if !defined(TECA_HAS_BACKTRACE)
        "WARNING: the stack could not be examined "
        "because backtrace is not supported.\n"
#elif !defined(TECA_HAS_DEBUG_BUILD)
        "WARNING: the stack trace will not use advanced "
        "capabilities because this is a release build.\n"
#else
#if !defined(TECA_HAS_SYMBOL_LOOKUP)
        "WARNING: function names will not be demangled because "
        "dladdr is not available.\n"
#endif
#if !defined(TECA_HAS_CPP_DEMANGLE)
        "WARNING: function names will not be demangled "
        "because cxxabi is not available.\n"
#endif
#endif
        ;

    std::ostringstream oss;
#if defined(TECA_HAS_BACKTRACE)
    void *stack_symbols[256];
    int n_frames = backtrace(stack_symbols, 256);
    for (int i = first_frame; i < n_frames; ++i)
    {
        symbol_properties sym_props;
        sym_props.set_report_path(whole_path);
        sym_props.initialize(stack_symbols[i]);
        oss << sym_props << std::endl;
    }
#else
    (void)first_frame;
    (void)whole_path;
#endif
    program_stack += oss.str();
    return program_stack;
}

// **************************************************************************
void set_stack_trace_on_error(int enable)
{
#if !defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)
    static int sa_orig_valid = 0;
    static struct sigaction sa_abrt_orig;
    static struct sigaction sa_segv_orig;
    static struct sigaction sa_term_orig;
    static struct sigaction sa_int_orig;
    static struct sigaction sa_ill_orig;
    static struct sigaction sa_bus_orig;
    static struct sigaction sa_fpe_orig;

    if (enable && !sa_orig_valid)
    {
        // save the current actions
        sigaction(SIGABRT, 0, &sa_abrt_orig);
        sigaction(SIGSEGV, 0, &sa_segv_orig);
        sigaction(SIGTERM, 0, &sa_term_orig);
        sigaction(SIGINT, 0, &sa_int_orig);
        sigaction(SIGILL, 0, &sa_ill_orig);
        sigaction(SIGBUS, 0, &sa_bus_orig);
        sigaction(SIGFPE, 0, &sa_fpe_orig);

        // enable read,  disable write
        sa_orig_valid = 1;

        // install ours
        struct sigaction sa;
        sa.sa_sigaction = (sig_action_t)stacktrace_signal_handler;
        sa.sa_flags = SA_SIGINFO | SA_RESETHAND;
# ifdef SA_RESTART
        sa.sa_flags |= SA_RESTART;
# endif
        sigemptyset(&sa.sa_mask);

        sigaction(SIGABRT, &sa, 0);
        sigaction(SIGSEGV, &sa, 0);
        sigaction(SIGTERM, &sa, 0);
        sigaction(SIGINT, &sa, 0);
        sigaction(SIGILL, &sa, 0);
        sigaction(SIGBUS, &sa, 0);
        sigaction(SIGFPE, &sa, 0);
    }
    else
    if (!enable && sa_orig_valid)
    {
        // restore previous actions
        sigaction(SIGABRT, &sa_abrt_orig, 0);
        sigaction(SIGSEGV, &sa_segv_orig, 0);
        sigaction(SIGTERM, &sa_term_orig, 0);
        sigaction(SIGINT, &sa_int_orig, 0);
        sigaction(SIGILL, &sa_ill_orig, 0);
        sigaction(SIGBUS, &sa_bus_orig, 0);
        sigaction(SIGFPE, &sa_fpe_orig, 0);

        // enable write,  disable read
        sa_orig_valid = 0;
    }
#else
    // avoid warning C4100
    (void)enable;
#endif
}

};
