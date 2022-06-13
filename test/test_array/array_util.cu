#include "array_util.h"

namespace array_util
{
// **************************************************************************
p_array cuda_accessible(const p_array &a)
{
    if (a->cuda_accessible())
        return a;

    std::cerr
        << "array_util::cuda_accessible moving data to the GPU" << std::endl;

    p_array tmp = array::new_cuda_accessible();
    tmp->copy(a);

    return tmp;
}

// **************************************************************************
const_p_array cuda_accessible(const const_p_array &a)
{
    if (a->cuda_accessible())
        return a;

    std::cerr
        << "array_util::cuda_accessible moving data to the GPU" << std::endl;

    p_array tmp = array::new_cuda_accessible();
    tmp->copy(a);

    return tmp;
}
}
