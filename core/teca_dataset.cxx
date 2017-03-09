#include "teca_dataset.h"
#include "teca_metadata.h"

// --------------------------------------------------------------------------
teca_dataset::teca_dataset()
{
    this->metadata = new teca_metadata;
}

// --------------------------------------------------------------------------
teca_dataset::~teca_dataset()
{
    delete this->metadata;
}

// --------------------------------------------------------------------------
void teca_dataset::copy(const const_p_teca_dataset &other)
{
    *this->metadata = *(other->metadata);
}

// --------------------------------------------------------------------------
void teca_dataset::shallow_copy(const p_teca_dataset &other)
{
    *this->metadata = *(other->metadata);
}

// --------------------------------------------------------------------------
void teca_dataset::swap(p_teca_dataset &other)
{
    teca_metadata *tmp = this->metadata;
    this->metadata = other->metadata;
    other->metadata = tmp;
}

// --------------------------------------------------------------------------
void teca_dataset::copy_metadata(const const_p_teca_dataset &other)
{
    *this->metadata = *(other->metadata);
}

// --------------------------------------------------------------------------
teca_metadata &teca_dataset::get_metadata() noexcept
{
    return *this->metadata;
}

// --------------------------------------------------------------------------
const teca_metadata &teca_dataset::get_metadata() const noexcept
{
    return *this->metadata;
}

// --------------------------------------------------------------------------
void teca_dataset::set_metadata(const teca_metadata &md)
{
    *this->metadata = md;
}

// --------------------------------------------------------------------------
void teca_dataset::to_stream(teca_binary_stream &bs) const
{
    this->metadata->to_stream(bs);
}

// --------------------------------------------------------------------------
void teca_dataset::from_stream(teca_binary_stream &bs)
{
    this->metadata->from_stream(bs);
}

// --------------------------------------------------------------------------
void teca_dataset::to_stream(std::ostream &os) const
{
    this->metadata->to_stream(os);
}

// --------------------------------------------------------------------------
void teca_dataset::from_stream(std::istream &)
{}
