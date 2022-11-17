#ifndef teca_variant_array_operator
#define teca_variant_array_operator

/// @file

#include "teca_variant_array.h"
#include "teca_type_select.h"

/// Codes dealing with run time specified operations on teca_variant_arrays
namespace teca_variant_array_operator
{
/// @cond
namespace internal
{
// --------------------------------------------------------------------------
template <typename nt_arg1, typename nt_arg2, typename nt_arg3,
    typename operator_t, typename type_select_t =
        typename teca_type_select::elevate<nt_arg2, nt_arg3>>
p_teca_variant_array apply(unsigned long n,
    const nt_arg1 *parg1, const nt_arg2 *parg2, const nt_arg3 *parg3,
    const operator_t &op)
{
    using nt_out = typename type_select_t::type;

    p_teca_variant_array_impl<nt_out> out =
        teca_variant_array_impl<nt_out>::New(n);

    auto spout = out->get_cpu_accessible();
    nt_out *pout = spout.get();

    for (unsigned long i = 0; i < n; ++i)
        pout[i] = static_cast<nt_out>(op(parg1[i], parg2[i], parg3[i]));

    return out;
}

// --------------------------------------------------------------------------
template <typename nt_larg, typename nt_rarg, typename operator_t,
    typename type_select_t = typename teca_type_select::elevate<nt_larg, nt_rarg>>
p_teca_variant_array apply(unsigned long n,
    const nt_larg *plarg, const nt_rarg *prarg, const operator_t &op)
{
    using nt_out = typename type_select_t::type;

    p_teca_variant_array_impl<nt_out> out =
        teca_variant_array_impl<nt_out>::New(n);

    auto spout = out->get_cpu_accessible();
    nt_out *pout = spout.get();

    for (unsigned long i = 0; i < n; ++i)
        pout[i] = static_cast<nt_out>(op(plarg[i], prarg[i]));

    return out;
}

// --------------------------------------------------------------------------
template <typename nt_arg, typename operator_t>
p_teca_variant_array apply(unsigned long n,
    const nt_arg *parg, const operator_t &op)
{
    p_teca_variant_array_impl<nt_arg> out =
        teca_variant_array_impl<nt_arg>::New(n);

    auto spout = out->get_cpu_accessible();
    nt_arg *pout = spout.get();

    for (unsigned long i = 0; i < n; ++i)
        pout[i] = static_cast<nt_arg>(op(parg[i]));

    return out;
}
};
/// @endcond

// --------------------------------------------------------------------------
template <typename operator_t>
p_teca_variant_array apply(const const_p_teca_variant_array &arg1,
    const const_p_teca_variant_array &arg2, const const_p_teca_variant_array &arg3,
    const operator_t &op)
{
    NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
        arg1.get(), _1,
        auto sparg1 = static_cast<const TT_1*>(arg1.get())->get_cpu_accessible();
        auto parg1 = sparg1.get();
        NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
            arg2.get(), _2,
            auto sparg2 = static_cast<const TT_2*>(arg2.get())->get_cpu_accessible();
            auto parg2 = sparg2.get();
            NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
                arg2.get(), _3,
                auto sparg3 = static_cast<const TT_3*>(arg3.get())->get_cpu_accessible();
                auto parg3 = sparg3.get();
                return internal::apply(arg1->size(), parg1, parg2, parg3, op);
                )
            )
        )
    TECA_ERROR("failed to apply " << operator_t::name() << ". unsupported type.")
    return nullptr;
}

// --------------------------------------------------------------------------
template <typename operator_t>
p_teca_variant_array apply_i(const const_p_teca_variant_array &larg,
    const const_p_teca_variant_array &rarg, const operator_t &op)
{
    NESTED_TEMPLATE_DISPATCH_I(teca_variant_array_impl,
        larg.get(), _LEFT,
        auto splarg = static_cast<const TT_LEFT*>(larg.get())->get_cpu_accessible();
        auto plarg = splarg.get();
        NESTED_TEMPLATE_DISPATCH_I(teca_variant_array_impl,
            rarg.get(), _RIGHT,
            auto sprarg = static_cast<const TT_RIGHT*>(rarg.get())->get_cpu_accessible();
            auto prarg = sprarg.get();
            return internal::apply(larg->size(), plarg, prarg, op);
            )
        )
    TECA_ERROR("failed to apply " << operator_t::name() << ". unsupported type.")
    return nullptr;
}

// --------------------------------------------------------------------------
template <typename operator_t>
p_teca_variant_array apply(const const_p_teca_variant_array &larg,
    const const_p_teca_variant_array &rarg, const operator_t &op)
{
    NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
        larg.get(), _LEFT,
        auto splarg = static_cast<const TT_LEFT*>(larg.get())->get_cpu_accessible();
        auto plarg = splarg.get();
        NESTED_TEMPLATE_DISPATCH(teca_variant_array_impl,
            rarg.get(), _RIGHT,
            auto sprarg = static_cast<const TT_RIGHT*>(rarg.get())->get_cpu_accessible();
            auto prarg = sprarg.get();
            return internal::apply(larg->size(), plarg, prarg, op);
            )
        )
    TECA_ERROR("failed to apply " << operator_t::name() << ". unsupported type.")
    return nullptr;
}

// --------------------------------------------------------------------------
template <typename operator_t>
p_teca_variant_array apply(const const_p_teca_variant_array &arg,
    const operator_t &op)
{
    TEMPLATE_DISPATCH(teca_variant_array_impl,
        arg.get(),
        auto sparg = static_cast<const TT*>(arg.get())->get_cpu_accessible();
        auto parg = sparg.get();
        return internal::apply(arg->size(), parg, op);
        )
    TECA_ERROR("failed to apply " << operator_t::name() << ". unsupported type.")
    return nullptr;
}

struct ternary_condition
{
template<typename nt_arg1, typename nt_arg2, typename nt_arg3,
typename nt_out = typename teca_type_select::elevate<nt_arg2, nt_arg3>::type>
nt_out operator()(const nt_arg1 &arg1, const nt_arg2 &arg2, const nt_arg3 &arg3) const
{ return (arg1 ? static_cast<nt_out>(arg2) : static_cast<nt_out>(arg3)); }
static constexpr const char *name(){ return "ternary_condition"; }
};

#define binary_operator(_name, _op)                                             \
struct _name                                                                    \
{                                                                               \
template<typename nt_larg, typename nt_rarg,                                    \
typename nt_out = typename teca_type_select::elevate<nt_larg, nt_rarg>::type>   \
nt_out operator()(const nt_larg &larg, const nt_rarg &rarg) const               \
{ return static_cast<nt_out>(larg) _op static_cast<nt_out>(rarg); }             \
static constexpr const char *name(){ return #_op; }                             \
};

binary_operator(add, +)
binary_operator(subtract, -)
binary_operator(multiply, *)
binary_operator(divide, /)
binary_operator(modulo, %)
binary_operator(logical_and, &&)
binary_operator(logical_or, ||)
binary_operator(less, <)
binary_operator(less_equal, <=)
binary_operator(greater, >)
binary_operator(greater_equal, >=)
binary_operator(equal, ==)
binary_operator(not_equal, !=)

#define binary_operator_fun(_name, _op)                                         \
struct _name                                                                    \
{                                                                               \
template<typename nt_larg, typename nt_rarg,                                    \
typename nt_out = typename teca_type_select::elevate<nt_larg, nt_rarg>::type>   \
nt_out operator()(const nt_larg &larg, const nt_rarg &rarg) const               \
{ return _op(static_cast<nt_out>(larg), static_cast<nt_out>(rarg)); }           \
static constexpr const char *name(){ return #_op; }                             \
};

binary_operator_fun(power, std::pow)

#define unary_operator(_name, _op)                      \
struct _name                                            \
{                                                       \
template<typename nt_arg>                               \
nt_arg operator()(const nt_arg &arg) const              \
{ return static_cast<nt_arg>(_op arg); }                \
static constexpr const char *name(){ return #_op; }     \
};

unary_operator(logical_not, !)

struct resolver
{
    static int invoke(const char *op, p_teca_variant_array &r,
        const const_p_teca_variant_array &a1, const const_p_teca_variant_array &a2,
        const const_p_teca_variant_array &a3)
    {
        switch (*op)
        {
            case '?': r = apply(a1, a2, a3, ternary_condition()); return 0;
        }
        r = nullptr;
        TECA_ERROR("binary operation " << op << " is not implemented")
        return -1;
    }
    static int invoke(const char *op, p_teca_variant_array &r,
        const const_p_teca_variant_array &a1, const const_p_teca_variant_array &a2)
    {
        switch (*op)
        {
            case '=': r = apply(a1, a2, equal()); return 0;
            case '!': r = apply(a1, a2, not_equal()); return 0;
            case '<':
                if (op[1] == '=') r = apply(a1, a2, less_equal());
                else r = apply(a1, a2, less());
                return 0;
            case '>':
                if (op[1] == '=') r = apply(a1, a2, greater_equal());
                else r = apply(a1, a2, greater());
                return 0;
            case '&': r = apply(a1, a2, logical_and()); return 0;
            case '|': r = apply(a1, a2, logical_or()); return 0;
            case '+': r = apply(a1, a2, add()); return 0;
            case '-': r = apply(a1, a2, subtract()); return 0;
            case '*':
                if (op[1] == '*') r = apply(a1, a2, power());
                else r = apply(a1, a2, multiply());
                return 0;
            case '/': r = apply(a1, a2, divide()); return 0;
            case '%': r = apply_i(a1, a2, modulo()); return 0;

        }
        r = nullptr;
        TECA_ERROR("binary operation " << op << " is not implemented")
        return -1;
    }

    static int invoke(const char *op, p_teca_variant_array &r,
        const const_p_teca_variant_array &a1)
    {
        switch (*op)
        {
            case '!':
                r = teca_variant_array_operator::apply(a1, logical_not());
                return 0;
                break;
        }
        r = nullptr;
        TECA_ERROR("unary operation " << op << " is not implemented")
        return -1;
    }

};
};

#endif
