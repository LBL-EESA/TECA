#ifndef teca_variant_array_util_h
#define teca_variant_array_util_h

/// @file

#include <tuple>

#if defined(TECA_HAS_CUDA)
#include "teca_cuda_util.h"
#endif

/// some functions helping us manipulate teca_variant_array
namespace teca_variant_array_util
{
/// synchronize the default stream.
inline
void synchronize_stream()
{
#if defined(TECA_HAS_CUDA)
    teca_cuda_util::synchronize_stream();
#endif
}

/** synchronize the default stream once if any of the passed arrays are not
 * accessible on the host. this should be done after all get_host_accessible
 * are issued and before the data is accessed.
 */
template <typename... array_t>
void sync_host_access_any(const array_t &... arrays)
{
    if ((!arrays->host_accessible() || ...))
    {
#if defined(TECA_HAS_CUDA)
        teca_cuda_util::synchronize_stream();
#endif
    }
}

/** static_cast a number of p_teca_variant_array into their derived type
 * teca_variant_array_impl<NT>*. Can be used with p_const_teca_variant_array as
 * well.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @tparam PP a parameter pack
 * @param args some number of p_teca_variant_array instances.
 * @returns a std::tuple of teca_variant_array_impl<NT>* one for each of args
 *          in the same order.
 */
template <typename TT, typename... PP>
auto va_static_cast(PP &&... args)
{
    return std::make_tuple(static_cast<TT*>(args.get())...);
}

/** dynamic_cast a number of p_teca_variant_array into their derived type
 * teca_variant_array_impl<NT>*. Can be used with p_const_teca_variant_array as
 * well.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @tparam PP a parameter pack
 * @param args some number of p_teca_variant_array instances.
 * @returns a std::tuple of teca_variant_array_impl<NT>* one for each of args
 *          in the same order.
 */
template <typename TT, typename... PP>
auto va_dynamic_cast(PP &&... args)
{
    return std::make_tuple(dynamic_cast<TT*>(args.get())...);
}

/// terminates recursion
template <typename TT>
void assert_type() {}

/** Check that a number of p_teca_variant_array are teca_variant_array_impl<NT>
 * instances. This will terminate execution if the check fails. Unlike asssert
 * this check is always applied.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @tparam PP a parameter pack
 * @param va a p_teca_variant_array instance
 * @param args some number of p_teca_variant_array instances.
 */
template <typename TT, typename... PP>
void assert_type(const const_p_teca_variant_array &va, PP &&... args)
{
    if (!dynamic_cast<const TT*>(va.get()))
    {
        TECA_FATAL_ERROR("teca_variant_array instance is not a"
            " teca_variant_array_impl<" << typeid(typename TT::element_type).name()
            << sizeof(typename TT::element_type) << ">")
    }

    (assert_type<TT>(args), ...);
}

/// terminates recursion
template <typename TT>
auto get_host_accessible()
{
    return std::make_tuple();
}

/** Calls teca_varaint_array_impl<NT>::get_host_accessible on a number of
 * p_teca_variant_array instances. The instances are first static_cast to
 * teca_variant_array_impl<NT>*. One should only use this method when one is
 * certain that this static_cast is appropriate. See va_assert_type for one
 * way to validate that the types are as expected.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @tparam PP a paramater pack
 * @param va a p_teca_variant_array instance
 * @param args any number of p_teca_variant_array instances
 * @returns a tuple of std::shared_ptr<NT> and NT* one for each
 *          p_teca_variant_array passed in.
 */
template <typename TT, typename... PP>
auto get_host_accessible(const std::shared_ptr<const TT> &va, PP &&... args)
{
    auto tva = static_cast<const TT*>(va.get());
    auto spva = tva->get_host_accessible();
    return std::tuple_cat(std::make_tuple
        (spva, spva.get()), get_host_accessible<TT>(args...));
}

/** Calls teca_varaint_array_impl<NT>::get_host_accessible on a number of
 * p_teca_variant_array instances. The instances are first static_cast to
 * teca_variant_array_impl<NT>*. One should only use this method when one is
 * certain that this static_cast is appropriate. See va_assert_type for one
 * way to validate that the types are as expected.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @tparam PP a paramater pack
 * @param va a const_p_teca_variant_array instance
 * @param args any number of p_teca_variant_array instances
 * @returns a tuple of std::shared_ptr<const NT> and const NT* one for each
 *          p_teca_variant_array passed in.
 */
template <typename TT, typename... V>
auto get_host_accessible(const std::shared_ptr<TT> &va, V &&... args)
{
    auto tva = static_cast<TT*>(va.get());
    auto spva = tva->get_host_accessible();
    return std::tuple_cat(std::make_tuple
        (spva, spva.get()), get_host_accessible<TT>(args...));
}

/** Calls teca_varaint_array_impl<NT>::get_host_accessible on a number of
 * p_teca_variant_array instances. The instances are first static_cast to
 * teca_variant_array_impl<NT>*. One should only use this method when one is
 * certain that this static_cast is appropriate. See va_assert_type for one
 * way to validate that the types are as expected.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @tparam PP a paramater pack
 * @param va a p_teca_variant_array instance
 * @param args any number of p_teca_variant_array instances
 * @returns a tuple of std::shared_ptr<NT> and NT* one for each
 *          p_teca_variant_array passed in.
 */
template <typename TT, typename... PP>
auto get_host_accessible(const const_p_teca_variant_array &va, PP &&... args)
{
    auto tva = static_cast<const TT*>(va.get());
    auto spva = tva->get_host_accessible();
    return std::tuple_cat(std::make_tuple
        (spva, spva.get()), get_host_accessible<TT>(args...));
}

/** Calls teca_varaint_array_impl<NT>::get_host_accessible on a number of
 * p_teca_variant_array instances. The instances are first static_cast to
 * teca_variant_array_impl<NT>*. One should only use this method when one is
 * certain that this static_cast is appropriate. See va_assert_type for one
 * way to validate that the types are as expected.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @tparam PP a paramater pack
 * @param va a const_p_teca_variant_array instance
 * @param args any number of p_teca_variant_array instances
 * @returns a tuple of std::shared_ptr<const NT> and const NT* one for each
 *          p_teca_variant_array passed in.
 */
template <typename TT, typename... V>
auto get_host_accessible(const p_teca_variant_array &va, V &&... args)
{
    auto tva = static_cast<TT*>(va.get());
    auto spva = tva->get_host_accessible();
    return std::tuple_cat(std::make_tuple
        (spva, spva.get()), get_host_accessible<TT>(args...));
}

/// terminates recursion
template <typename TT>
auto get_cuda_accessible()
{
    return std::make_tuple();
}

/** Calls teca_varaint_array_impl<NT>::get_cuda_accessible on a number of
 * p_teca_variant_array instances. The instances are first static_cast to
 * teca_variant_array_impl<NT>*. One should only use this method when one is
 * certain that this static_cast is appropriate. See va_assert_type for one
 * way to validate that the types are as expected.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @tparam PP a paramater pack
 * @param va a p_teca_variant_array instance
 * @param args any number of p_teca_variant_array instances
 * @returns a tuple of std::shared_ptr<NT> and NT* one for each
 *          p_teca_variant_array passed in.
 */
template <typename TT, typename... PP>
auto get_cuda_accessible(const const_p_teca_variant_array &va, PP &&... args)
{
    auto tva = static_cast<const TT*>(va.get());
    auto spva = tva->get_cuda_accessible();
    return std::tuple_cat(std::make_tuple
        (spva, spva.get()), get_cuda_accessible<TT>(args...));
}

/** Calls teca_varaint_array_impl<NT>::get_cuda_accessible on a number of
 * p_teca_variant_array instances. The instances are first static_cast to
 * teca_variant_array_impl<NT>*. One should only use this method when one is
 * certain that this static_cast is appropriate. See va_assert_type for one
 * way to validate that the types are as expected.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @tparam PP a paramater pack
 * @param va a const_p_teca_variant_array instance
 * @param args any number of p_teca_variant_array instances
 * @returns a tuple of std::shared_ptr<const NT> and const NT* one for each
 *          p_teca_variant_array passed in.
 */
template <typename TT, typename... V>
auto get_cuda_accessible(const p_teca_variant_array &va, V &&... args)
{
    auto tva = static_cast<TT*>(va.get());
    auto spva = tva->get_cuda_accessible();
    return std::tuple_cat(std::make_tuple
        (spva, spva.get()), get_cuda_accessible<TT>(args...));
}

/** Calls teca_varaint_array_impl<NT>::data on a number of
 * p_teca_variant_array instances. The instances are first static_cast to
 * teca_variant_array_impl<NT>*. One should only use this method when one is
 * certain that this static_cast is appropriate. See va_assert_type for one
 * way to validate that the types are as expected.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @tparam PP a paramater pack
 * @param args any number of p_teca_variant_array instances
 * @returns a tuple of NT* one for each p_teca_variant_array passed in.
 */
template <typename TT, typename... V>
auto data(V &&... args)
{
    return std::make_tuple(static_cast<TT*>(args.get())->data()...);
}

/** Allocates a teca_variant_array_impl<NT> instance and returns the newly
 * allocated array and a pointer to it's memory.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @param n_elem the size of the array
 * @param alloc the allocator to use
 * @returns a tuple of p_teca_variant_array_impl<NT> and NT*
 */
template <typename TT>
auto New(size_t n_elem,
    teca_variant_array::allocator alloc = teca_variant_array::allocator::malloc)
{
    auto out = TT::New(n_elem, alloc);
    return std::make_tuple(out, out->data());
}

/** Allocates a teca_variant_array_impl<NT> instance and returns the newly
 * allocated array and a pointer to it's memory.
 *
 * @tparam TT teca_variant_array_impl<NT>
 * @param n_elem the size of the array
 * @param init_val a value to initialize the contents of the array to
 * @param alloc the allocator to use
 * @returns a tuple of p_teca_variant_array_impl<NT> and NT*
 */
template <typename TT, typename NT = typename TT::element_type>
auto New(size_t n_elem, NT init_val,
    teca_variant_array::allocator alloc = teca_variant_array::allocator::malloc)
{
    auto out = TT::New(n_elem, init_val, alloc);
    return std::make_tuple(out, out->data());
}

}
#endif
