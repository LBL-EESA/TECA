#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_common.h"

// --------------------------------------------------------------------------
void teca_variant_array::copy(const teca_variant_array &other)
{
    TEMPLATE_DISPATCH_CLASS(
        teca_variant_array_impl, std::string, this, &other,
        *p1_tt = *p2_tt;
        return;
        )
    TEMPLATE_DISPATCH_CLASS(
        teca_variant_array_impl, teca_metadata, this, &other,
        *p1_tt = *p2_tt;
        return;
        )
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->copy(other);
        return;
        )
    throw std::bad_cast();
}

// --------------------------------------------------------------------------
void teca_variant_array::append(const teca_variant_array &other)
{
    TEMPLATE_DISPATCH_CLASS(
        teca_variant_array_impl, std::string, this, &other,
        p1_tt->append(p2_tt->m_data);
        return;
        )
    TEMPLATE_DISPATCH_CLASS(
        teca_variant_array_impl, teca_metadata, this, &other,
        p1_tt->append(p2_tt->m_data);
        return;
        )
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(other);
        return;
        )
    throw std::bad_cast();
}
