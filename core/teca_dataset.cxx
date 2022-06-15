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
int teca_dataset::set_request_index(const std::string &key, unsigned long val)
{
    unsigned long ids[2] = {val, val};
    return this->set_request_indices(key, ids);
}

// --------------------------------------------------------------------------
int teca_dataset::get_request_index(unsigned long &val) const
{
    std::string index_request_key;
    if (this->metadata->get("index_request_key", index_request_key))
    {
        TECA_ERROR("An index_request_key has not been set")
        return -1;
    }

    unsigned long ids[2] = {0};
    if (this->metadata->get(index_request_key, ids))
    {
        TECA_ERROR("Failed to get the value of the index_request_key \""
            << index_request_key <<"\"")
        return -1;
    }

    unsigned long n_ids = ids[1] - ids[0] + 1;
    if (n_ids != 1)
    {
        TECA_ERROR(<< n_ids << " indices requested but this algorithm"
            " handles only a single index")
        return -1;
    }

    val = ids[0];

    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset::set_request_indices(const std::string &key,
    const unsigned long val[2])
{
    if (this->metadata->set("index_request_key", key)  ||
        this->metadata->set(key, val, 2))
    {
        TECA_ERROR("failed to set the index_request_key \""
            << key << "\" to [" << val << "]")
        return -1;
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset::get_request_indices(unsigned long val[2]) const
{
    std::string index_request_key;
    if (this->metadata->get("index_request_key", index_request_key))
    {
        TECA_ERROR("An index_request_key has not been set")
        return -1;
    }

    return this->metadata->get(index_request_key, val);
}

// --------------------------------------------------------------------------
void teca_dataset::copy(const const_p_teca_dataset &other,
    teca_variant_array::allocator alloc)
{
    (void) alloc;
    *this->metadata = *(other->metadata);
}

// --------------------------------------------------------------------------
void teca_dataset::shallow_copy(const p_teca_dataset &other)
{
    *this->metadata = *(other->metadata);
}

// --------------------------------------------------------------------------
void teca_dataset::swap(const p_teca_dataset &other)
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
int teca_dataset::to_stream(teca_binary_stream &bs) const
{
    std::string class_name = this->get_class_name();
    bs.pack(class_name.c_str(), class_name.size());

    if (this->metadata->to_stream(bs))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset::from_stream(teca_binary_stream &bs)
{
    std::string class_name = this->get_class_name();

    if (bs.expect(class_name.c_str(), class_name.size()))
    {
        TECA_ERROR("invalid stream")
        return -1;
    }

    if (this->metadata->from_stream(bs))
    {
        TECA_ERROR("Failed to deserialize metadata")
        return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset::to_stream(std::ostream &os) const
{
    if (this->metadata->to_stream(os))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
int teca_dataset::from_stream(std::istream &)
{
    return -1;
}
