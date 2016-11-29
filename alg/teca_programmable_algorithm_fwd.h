#ifndef teca_program_algorithm_fwd_h
#define teca_program_algorithm_fwd_h

#include "teca_shared_object.h"
#include "teca_metadata.h"
#include "teca_dataset_fwd.h"

TECA_SHARED_OBJECT_FORWARD_DECL(teca_programmable_algorithm)

#ifdef SWIG
typedef void* report_callback_t;
typedef void* request_callback_t;
typedef void* execute_callback_t;
#else
using report_callback_t = std::function<teca_metadata(
        unsigned int, const std::vector<teca_metadata>&)>;

using request_callback_t = std::function<std::vector<teca_metadata>(
        unsigned int, const std::vector<teca_metadata> &,
        const teca_metadata &)>;

using execute_callback_t = std::function<const_p_teca_dataset(
        unsigned int, const std::vector<const_p_teca_dataset> &,
        const teca_metadata &)>;
#endif
#endif
