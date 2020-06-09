#include "teca_array_attributes.h"

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

    return 0;
}

// --------------------------------------------------------------------------
int teca_array_attributes::from(const teca_metadata &md)
{
    return  md.get("type_code", type_code) +
        md.get("centering", centering) +
        md.get("size", size) +
        md.get("units", units) +
        md.get("long_name", long_name) +
        md.get("description", description);
}

// --------------------------------------------------------------------------
void teca_array_attributes::to_stream(std::ostream &os)
{
    os << "type_code=" << type_code << ", centering=" << centering
        << ", size=" << size << ", units=\"" << units
        << "\", long_name=\"" << long_name << "\" description=\""
        << description << "\"";
}
