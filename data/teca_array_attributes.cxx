#include "teca_array_attributes.h"


using fill_value_t =
    std::variant<char, unsigned char, short, unsigned short,
         int, unsigned int, long, unsigned long, long long,
         unsigned long long, float, double>;

#define get_cast_case(_to_t, _from_t)                     \
    if (const _from_t *ptr = std::get_if<_from_t>(&fv))   \
    {                                                     \
        return (_to_t)*ptr;                               \
    }

template <typename to_t>
to_t get_cast(const fill_value_t &fv)
{
    get_cast_case(to_t, char)
    else get_cast_case(to_t, unsigned char)
    else get_cast_case(to_t, short)
    else get_cast_case(to_t, unsigned short)
    else get_cast_case(to_t, int)
    else get_cast_case(to_t, unsigned int)
    else get_cast_case(to_t, long)
    else get_cast_case(to_t, unsigned long)
    else get_cast_case(to_t, long long)
    else get_cast_case(to_t, unsigned long long)
    else get_cast_case(to_t, float)
    else get_cast_case(to_t, double)
    TECA_ERROR("bad fill_value type")
    return to_t();
}

// --------------------------------------------------------------------------
teca_array_attributes &teca_array_attributes::operator=(const teca_metadata &md)
{
    from(md);
    return *this;
}

// --------------------------------------------------------------------------
teca_array_attributes::operator teca_metadata() const
{
    teca_metadata atts;
    to(atts);
    return atts;
}

// --------------------------------------------------------------------------
int teca_array_attributes::to(teca_metadata &md) const
{
    if (type_code > 0)
        md.set("type_code", type_code);

    if (centering > 0)
        md.set("centering", centering);

    if (size > 0)
        md.set("size", size);

    if (!units.empty())
        md.set("units", units);

    if (!long_name.empty())
        md.set("long_name", long_name);

    if (!description.empty())
        md.set("description", description);

    if (have_fill_value)
    {
        if (type_code < 1)
        {
            TECA_ERROR("A valid type_code is required with a fill_value")
            return -1;
        }
        CODE_DISPATCH(type_code,
            md.set("_FillValue", get_cast<NT>(fill_value));
            )
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_array_attributes::merge_to(teca_metadata &md) const
{
    // always overwrite these, as they are internal and essential to
    // I/O opperations
    if (type_code > 0)
        md.set("type_code", type_code);

    if (centering > 0)
        md.set("centering", centering);

    if (size > 0)
        md.set("size", size);

    // preserve any existing values in these
    if (!units.empty() && !md.has("units"))
        md.set("units", units);

    if (!long_name.empty() && !md.has("long_name"))
        md.set("long_name", long_name);

    if (!description.empty() && !md.has("description"))
        md.set("description", description);

    if (have_fill_value && !md.has("_FillValue"))
    {
        if (type_code < 1)
        {
            TECA_ERROR("A valid type_code is required with a fill_value")
            return -1;
        }
        CODE_DISPATCH(type_code,
            md.set("_FillValue", get_cast<NT>(fill_value));
            )
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_array_attributes::from(const teca_metadata &md)
{
    md.get("type_code", type_code);
    md.get("centering", centering);
    md.get("size", size);
    md.get("units", units);
    md.get("long_name", long_name);
    md.get("description", description);

    if (type_code > 0)
    {
        CODE_DISPATCH(type_code,
            NT tmp = NT();
            have_fill_value = md.get("_FillValue", tmp) == 0;
            if (have_fill_value)
                fill_value = tmp;
            )
    }
    else
    {
        have_fill_value = 0;
    }

    return 0;
}

// --------------------------------------------------------------------------
void teca_array_attributes::to_stream(std::ostream &os) const
{
    os << "type_code=" << type_code << ", centering=" << centering
        << ", size=" << size << ", units=\"" << units
        << "\", long_name=\"" << long_name << "\" description=\""
        << description << "\", fill_value=";

    if (have_fill_value)
    {
        if (type_code < 1)
        {
            TECA_ERROR("A valid type_code is required with a fill_value")
        }
        CODE_DISPATCH(type_code,
            os << get_cast<NT>(fill_value);
            )
    }
    else
    {
        os << "None";
    }
}
