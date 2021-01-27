#include "teca_variant_array.h"
#include "teca_metadata.h"
#include "teca_common.h"

// --------------------------------------------------------------------------
void teca_variant_array::copy(const teca_variant_array &other)
{
    TEMPLATE_DISPATCH_CASE(
        teca_variant_array_impl, std::string, this,
        TT *p1_tt = static_cast<TT*>(this);
        const TT *p2_tt = dynamic_cast<const TT*>(&other);
        if (p2_tt)
        {
            *p1_tt = *p2_tt;
            return;
        }
        )
    else
    TEMPLATE_DISPATCH_CASE(
        teca_variant_array_impl, teca_metadata, this,
        TT *p1_tt = static_cast<TT*>(this);
        const TT *p2_tt = dynamic_cast<const TT*>(&other);
        if (p2_tt)
        {
            *p1_tt = *p2_tt;
            return;
        }
        )
    else
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->copy(other);
        return;
        )
    TECA_ERROR("Can't copy a \"" << other.get_class_name()
        << "\" to a \"" << this->get_class_name() << "\"")
    throw teca_bad_cast(other.get_class_name(), this->get_class_name());
}

// --------------------------------------------------------------------------
void teca_variant_array::append(const teca_variant_array &other)
{
    TEMPLATE_DISPATCH_CASE(
        teca_variant_array_impl, std::string, this,
        TT *p1_tt = static_cast<TT*>(this);
        const TT *p2_tt = dynamic_cast<const TT*>(&other);
        if (p2_tt)
        {
            p1_tt->append(p2_tt->m_data);
            return;
        }
        )
    else
    TEMPLATE_DISPATCH_CASE(
        teca_variant_array_impl, teca_metadata, this,
        const TT *p2_tt = dynamic_cast<const TT*>(&other);
        if (p2_tt)
        {
            TT *p1_tt = static_cast<TT*>(this);
            p1_tt->append(p2_tt->m_data);
            return;
        }
        )
    else
    TEMPLATE_DISPATCH(teca_variant_array_impl, this,
        TT *this_t = static_cast<TT*>(this);
        this_t->append(other);
        return;
        )
    TECA_ERROR("Can't append a \"" << other.get_class_name()
        << "\" to a \"" << this->get_class_name() << "\"")
    throw teca_bad_cast(other.get_class_name(), this->get_class_name());
}
