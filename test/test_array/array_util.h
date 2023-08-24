#ifndef array_util_h
#define array_util_h

#include "teca_config.h"
#include "array.h"

namespace array_util
{

/// ensure access on the CPU. Data is copied if needed.
TECA_EXPORT
p_array host_accessible(const p_array &a);

TECA_EXPORT
const_p_array host_accessible(const const_p_array &a);

/// ensure access from code CUDA code. Data is copied if needed.
TECA_EXPORT
p_array cuda_accessible(const p_array &a);

TECA_EXPORT
const_p_array cuda_accessible(const const_p_array &a);

}

#endif
