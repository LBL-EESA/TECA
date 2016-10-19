#ifndef teca_program_reduce_fwd_h
#define teca_program_reduce_fwd_h

#include "teca_shared_object.h"
#include "teca_dataset_fwd.h"
#include <functional>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_programmable_reduce)

#ifdef SWIG
typedef void* reduce_callback_t;
#else
using reduce_callback_t = std::function<p_teca_dataset(
    const const_p_teca_dataset &, const const_p_teca_dataset &)>;
#endif
#endif
