#include "teca_variant_array.h"
#include "teca_common.h"

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get(T &val, unsigned int i) const
{
    template_dispatch(const teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(val, i);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::get(std::string &val, unsigned int i) const
{
    template_dispatch_case(const teca_variant_array_impl<std::string>, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(val, i);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::get(std::vector<T> &vals) const
{
    template_dispatch(const teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::get(std::vector<std::string> &vals) const
{
    template_dispatch_case(const teca_variant_array_impl<std::string>, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->get(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set(const std::vector<T> &vals)
{
    template_dispatch(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::set(const std::vector<std::string> &vals)
{
    typedef teca_variant_array_impl<std::string> TT;
    TT *this_t = dynamic_cast<TT *>(this);
    if (this_t)
    {
        this_t->set(vals);
    }
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::set(const T &val, unsigned int i)
{
    template_dispatch(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(val, i);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::set(const std::string &val, unsigned int i)
{
    template_dispatch_case(teca_variant_array_impl<std::string>, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->set(val, i);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append(const T &val)
{
    template_dispatch(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(val);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::append(const std::string &val)
{
    template_dispatch_case(teca_variant_array_impl<std::string>, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(val);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
template<typename T>
void teca_variant_array::append(const std::vector<T> &vals)
{
    template_dispatch(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::append(const std::vector<std::string> &vals)
{
    template_dispatch_case(teca_variant_array_impl<std::string>, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(vals);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::copy(const teca_variant_array &other)
{
    typedef teca_variant_array_impl<std::string> STT;
    if (dynamic_cast<STT*>(this) && dynamic_cast<const STT*>(&other))
    {
        STT *this_t = static_cast<STT*>(this);
        const STT *other_t = static_cast<const STT*>(&other);
        *this_t = *other_t;
        return;
    }
    template_dispatch(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->copy(other);
        return;
        )
    throw std::bad_cast();
}
