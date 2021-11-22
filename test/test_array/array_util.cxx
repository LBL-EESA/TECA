#include "array_util.h"
#include "teca_config.h"

#if defined(TECA_HAS_CUDA)
namespace teca_cuda_util { int synchronize(); }
#endif

namespace array_util
{
// **************************************************************************
p_array cpu_accessible(const p_array &a)
{
    if (a->cpu_accessible())
        return a;

    std::cerr
        << "array_util::cpu_accessible moving data to the CPU" << std::endl;

#if defined(TECA_HAS_CUDA)
    if (a->cuda_accessible())
       teca_cuda_util::synchronize();
#endif

    p_array tmp = array::new_cpu_accessible();
    tmp->copy(a);

    return tmp;
}

// **************************************************************************
const_p_array cpu_accessible(const const_p_array &a)
{
    if (a->cpu_accessible())
        return a;

    std::cerr
        << "array_util::cpu_accessible moving data to the CPU" << std::endl;

#if defined(TECA_HAS_CUDA)
    if (a->cuda_accessible())
       teca_cuda_util::synchronize();
#endif

    p_array tmp = array::new_cpu_accessible();
    tmp->copy(a);

    return tmp;
}

#if !defined(TECA_HAS_CUDA)
// **************************************************************************
p_array cuda_accessible(const p_array &a)
{
    (void) a;
    TECA_ERROR("cuda_accessible failed because CUDA is unavailable")
    return nullptr;
}

// **************************************************************************
const_p_array cuda_accessible(const const_p_array &a)
{
    (void) a;
    TECA_ERROR("cuda_accessible failed because CUDA is unavailable")
    return nullptr;
}
#endif
}
