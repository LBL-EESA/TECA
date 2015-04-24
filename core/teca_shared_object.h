#ifndef teca_shared_object_h
#define teca_shared_object_h

#include <memory>

// convenience macro. every teca_algrotihm/dataset
// should have the following forward declarations
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
    using const_p_##_cls = std::shared_ptr<_cls<const T>>;

#endif
