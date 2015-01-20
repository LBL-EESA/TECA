#ifndef teca_temporal_reduction_fwd_h
#define teca_temporal_reduction_fwd_h

class teca_temporal_reduction;
typedef std::shared_ptr<teca_temporal_reduction> p_teca_temporal_reduction;

// this is a convenience macro to be used to declare a static
// New method that will be used to construct new objects in
// shared_ptr's. This manages the details of interoperability
// with std C++11 shared pointer
#define TECA_TEMPORAL_REDUCTION_STATIC_NEW(T)                       \
                                                                    \
static p_##T New()                                                  \
{                                                                   \
    return p_##T(new T);                                            \
}                                                                   \
                                                                    \
using                                                               \
enable_shared_from_this<teca_temporal_reduction>::shared_from_this; \
                                                                    \
std::shared_ptr<T> shared_from_this()                               \
{                                                                   \
    return std::static_pointer_cast<T>(shared_from_this());         \
}                                                                   \
                                                                    \
std::shared_ptr<T const> shared_from_this() const                   \
{                                                                   \
    return std::static_pointer_cast<T const>(shared_from_this());   \
}

#endif
