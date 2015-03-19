#include "teca_variant_array.h"
#include "teca_common.h"

// --------------------------------------------------------------------------
void teca_variant_array::get(std::string &val, unsigned long i) const
{
    TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl, std::string, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(val, i);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::get(std::vector<std::string> &vals) const
{
    TEMPLATE_DISPATCH_CASE(const teca_variant_array_impl, std::string, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::set(const std::vector<std::string> &vals)
{
    using TT = teca_variant_array_impl<std::string>;
    TT *this_t = dynamic_cast<TT *>(this);
    if (this_t)
    {
        this_t->set(vals);
    }
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::set(const std::string &val, unsigned long i)
{
    TEMPLATE_DISPATCH_CASE(teca_variant_array_impl, std::string, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(val, i);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::append(const std::string &val)
{
    TEMPLATE_DISPATCH_CASE(teca_variant_array_impl, std::string, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(val);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::append(const std::vector<std::string> &vals)
{
    TEMPLATE_DISPATCH_CASE(teca_variant_array_impl, std::string, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::copy(const teca_variant_array &other)
{
    using STT = teca_variant_array_impl<std::string>;
    if (dynamic_cast<STT*>(this) && dynamic_cast<const STT*>(&other))
    {
        STT *this_t = static_cast<STT*>(this);
        const STT *other_t = static_cast<const STT*>(&other);
        *this_t = *other_t;
        return;
    }
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->copy(other);
        return;
        )
    throw std::bad_cast();
}
