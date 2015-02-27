#ifndef teca_dataset_fwd_h
#define teca_dataset_fwd_h

#include <memory>

class teca_dataset;
typedef std::shared_ptr<teca_dataset> p_teca_dataset;

// this is a convenience macro to be used to
// declare New and enable seamless operation
// with std C++11 shared pointer
#define TECA_DATASET_STATIC_NEW(T)                                  \
                                                                    \
static p_##T New()                                                  \
{                                                                   \
    return p_##T(new T);                                            \
}                                                                   \
                                                                    \
using enable_shared_from_this<teca_dataset>::shared_from_this;      \
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

// convenience macro to expose default copy,
// shallow_copy and swap
#define TECA_DATASET_COPY_SWAP()        \
    using teca_dataset::swap;           \
    using teca_dataset::copy;           \
    using teca_dataset::shallow_copy;   \
    using teca_dataset::copy_metadata;

// convenience macro for adding properties to dataset
// objects
#define TECA_DATASET_PROPERTY(T, name)  \
                                        \
void set_##name(const T &val)           \
{                                       \
    this->name = val;                   \
}                                       \
                                        \
const T &get_##name() const             \
{                                       \
    return this->name;                  \
}                                       \
                                        \
T &get_##name()                         \
{                                       \
    return this->name;                  \
}

#endif
