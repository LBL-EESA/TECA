#ifndef array_util_h
#define array_util_h

#include "array.h"

namespace array_util
{

/// ensure access on the CPU. Data is copied if needed.
p_array cpu_accessible(const p_array &a);
const_p_array cpu_accessible(const const_p_array &a);

/// ensure access from code CUDA code. Data is copied if needed.
p_array cuda_accessible(const p_array &a);
const_p_array cuda_accessible(const const_p_array &a);

}

#endif
