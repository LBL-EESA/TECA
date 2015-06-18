#ifndef teca_program_source_fwd_h
#define teca_program_source_fwd_h

// helper that allows us to use std::function
// as a TECA_ALGORITHM_PROPERTY
template<typename T>
bool operator!=(const std::function<T> &lhs, const std::function<T> &rhs)
{
    return &rhs != &lhs;
}

// TODO -- this is a work around for older versions
// of Apple clang
// Apple LLVM version 4.2 (clang-425.0.28) (based on LLVM 3.2svn)
// Target: x86_64-apple-darwin12.6.0
#define TECA_PROGRAMMABLE_SOURCE_PROPERTY(T, NAME) \
                                         \
void set_##NAME(const T &v)              \
{                                        \
    /*if (this->NAME != v)*/                 \
    /*{*/                                    \
        this->NAME = v;                  \
        this->set_modified();            \
    /*}*/                                    \
}                                        \
                                         \
const T &get_##NAME() const              \
{                                        \
    return this->NAME;                   \
}                                        \
                                         \
T &get_##NAME()                          \
{                                        \
    return this->NAME;                   \
}

#endif
