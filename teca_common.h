#ifndef teca_common_h
#define teca_common_h

#define template_dispatch_case(t, p, body)  \
    if (dynamic_cast<t*>(p))                \
    {                                       \
        typedef t TT;                       \
        body                                \
    }

// macro for helping downcast to POD types
// don't add classes to this.
#define template_dispatch(t, p, body)                   \
    template_dispatch_case(t<char>, p, body)            \
    template_dispatch_case(t<unsigned char>, p, body)   \
    template_dispatch_case(t<int>, p, body)             \
    template_dispatch_case(t<unsigned int>, p, body)    \
    template_dispatch_case(t<long>, p, body)            \
    template_dispatch_case(t<unsigned long>, p, body)   \
    template_dispatch_case(t<float>, p, body)           \
    template_dispatch_case(t<double>, p, body)

#endif
