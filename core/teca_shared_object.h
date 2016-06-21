#ifndef teca_shared_object_h
#define teca_shared_object_h

#include <memory>

// convenience macro. every teca_algrotihm/dataset
// should have the following forward declarations
#ifdef SWIG
// SWIG doesn't handle alias templates yet. OK for the
// shared object forward but the shared object template
// forward has no direct mapping into c++03.
#define TECA_SHARED_OBJECT_FORWARD_DECL(_cls)           \
    class _cls;                                         \
    typedef std::shared_ptr<_cls> p_##_cls;             \
    typedef std::shared_ptr<const _cls> const_p_##_cls;

#define TECA_SHARED_OBJECT_TEMPLATE_FORWARD_DECL(_cls)  \
    template<typename T> class _cls;
#else
#define TECA_SHARED_OBJECT_FORWARD_DECL(_cls)           \
    class _cls;                                         \
    using p_##_cls = std::shared_ptr<_cls>;             \
    using const_p_##_cls = std::shared_ptr<const _cls>;

#define TECA_SHARED_OBJECT_TEMPLATE_FORWARD_DECL(_cls)  \
    template<typename T> class _cls;                    \
                                                        \
    template<typename T>                                \
    using p_##_cls = std::shared_ptr<_cls<T>>;          \
                                                        \
    template<typename T>                                \
    using const_p_##_cls = std::shared_ptr<const _cls<T>>;
#endif
#endif
