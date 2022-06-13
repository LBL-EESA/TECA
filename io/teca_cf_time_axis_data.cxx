#include "teca_cf_time_axis_data.h"

#include "teca_binary_stream.h"
#include "teca_dataset_util.h"
#include "teca_string_util.h"
#include "teca_bad_cast.h"

#include <iostream>
#include <iomanip>

// some helpers to disambiguate nested pairs
namespace {
template <typename it_t>
unsigned long get_file_id(const it_t &it)
{
    return it->first;
}

template <typename it_t>
teca_metadata &get_metadata(const it_t &it)
{
    return it->second.second;
}

template <typename it_t>
const teca_metadata &get_const_metadata(const it_t &it)
{
    return it->second.second;
}


template <typename it_t>
p_teca_variant_array get_variant_array(const it_t &it)
{
    return it->second.first;
}

template <typename it_t>
const_p_teca_variant_array get_const_variant_array(const it_t &it)
{
    return it->second.first;
}
};



// --------------------------------------------------------------------------
teca_cf_time_axis_data::teca_cf_time_axis_data()
{
    this->internals = new internals_t;
}

// --------------------------------------------------------------------------
teca_cf_time_axis_data::~teca_cf_time_axis_data()
{
    delete this->internals;
}

// --------------------------------------------------------------------------
void teca_cf_time_axis_data::transfer(unsigned long file_id, elem_t &&data)
{
    this->internals->emplace(file_id, std::forward<elem_t>(data));
}

// --------------------------------------------------------------------------
teca_cf_time_axis_data::elem_t &teca_cf_time_axis_data::get(unsigned long file_id)
{
    internals_t::iterator it = this->internals->find(file_id);
    if (it == this->internals->end())
    {
        TECA_ERROR("invalid file_id " << file_id)
    }
    return it->second;
}

// --------------------------------------------------------------------------
const teca_cf_time_axis_data::elem_t &teca_cf_time_axis_data::get(unsigned long file_id) const
{
    return const_cast<teca_cf_time_axis_data*>(this)->get(file_id);
}

// --------------------------------------------------------------------------
void teca_cf_time_axis_data::append(const const_p_teca_dataset &o)
{
    const_p_teca_cf_time_axis_data other =
        std::dynamic_pointer_cast<const teca_cf_time_axis_data>(o);

    if (!other)
        throw teca_bad_cast(safe_class_name(o), "teca_cf_time_axis_data");

    if (this == other.get())
        return;

    internals_t::const_iterator it = other->internals->begin();
    internals_t::const_iterator end = other->internals->end();
    for (; it != end; ++it)
    {
        p_teca_variant_array va = ::get_variant_array(it) ?
            ::get_variant_array(it)->new_copy() : nullptr;

        if (!this->internals->emplace(std::make_pair(::get_file_id(it),
            std::make_pair(va, ::get_const_metadata(it)))).second)
        {
            TECA_ERROR("file_id " << ::get_file_id(it) << " is not unique");
            return;
        }
    }
}

// --------------------------------------------------------------------------
void teca_cf_time_axis_data::shallow_append(const const_p_teca_dataset &o)
{
    const_p_teca_cf_time_axis_data other =
        std::dynamic_pointer_cast<const teca_cf_time_axis_data>(o);

    if (!other)
        throw teca_bad_cast(safe_class_name(o), "teca_cf_time_axis_data");

    if (this == other.get())
        return;

    internals_t::const_iterator it = other->internals->begin();
    internals_t::const_iterator end = other->internals->end();
    for (; it != end; ++it)
    {
        if (!this->internals->insert(*it).second)
        {
            TECA_ERROR("file_id " << ::get_file_id(it) << " is not unique");
            return;
        }
    }
}

// --------------------------------------------------------------------------
int teca_cf_time_axis_data::get_type_code() const
{
    // FIXME -- need a user defined index
    return (int)1e3;
}

// --------------------------------------------------------------------------
bool teca_cf_time_axis_data::empty() const noexcept
{
    return this->internals->empty();
}

// --------------------------------------------------------------------------
int teca_cf_time_axis_data::to_stream(teca_binary_stream &bs) const
{
    if (this->teca_dataset::to_stream(bs))
        return -1;

    unsigned long n_elem = this->internals->size();
    bs.pack(n_elem);

    internals_t::const_iterator it = this->internals->begin();
    internals_t::const_iterator end = this->internals->end();
    for (; it != end; ++it)
    {
        bs.pack(::get_file_id(it));

        bs.pack(::get_const_variant_array(it)->type_code());

        if (::get_const_variant_array(it)->to_stream(bs))
            return -1;

        if (::get_const_metadata(it).to_stream(bs))
            return -1;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_time_axis_data::from_stream(teca_binary_stream &bs)
{
    this->internals->clear();

    if (this->teca_dataset::from_stream(bs))
        return -1;

    unsigned long n_elem = 0;
    bs.unpack(n_elem);

    for (unsigned long i = 0; i < n_elem; ++i)
    {
        unsigned long file_id = 0;
        bs.unpack(file_id);

        unsigned int type_code;
        bs.unpack(type_code);

        p_teca_variant_array va = teca_variant_array_factory::New(type_code);

        if (va->from_stream(bs))
            return -1;

        teca_metadata md;
        if (md.from_stream(bs))
            return -1;

        this->internals->emplace(std::make_pair(file_id, std::make_pair(va, md)));
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_time_axis_data::to_stream(std::ostream &os) const
{
    internals_t::const_iterator it = this->internals->begin();
    internals_t::const_iterator end = this->internals->end();
    for (; it != end; ++it)
    {
        std::cerr << std::setw(6) << ::get_file_id(it) << " ";

        ::get_const_variant_array(it)->to_stream(os);

        os << std::endl;
    }
    return 0;
}

// --------------------------------------------------------------------------
int teca_cf_time_axis_data::from_stream(std::istream &)
{
    TECA_ERROR("teca_cf_time_axis_data::from_stream not implemented")
    return -1;
}

// --------------------------------------------------------------------------
void teca_cf_time_axis_data::copy(const const_p_teca_dataset &o,
    allocator alloc)
{
    const_p_teca_cf_time_axis_data other =
        std::dynamic_pointer_cast<const teca_cf_time_axis_data>(o);

    if (!other)
        throw teca_bad_cast(safe_class_name(o), "teca_cf_time_axis_data");

    if (this == other.get())
        return;

    this->internals->clear();

    internals_t::const_iterator it = other->internals->begin();
    internals_t::const_iterator end = other->internals->end();
    for (; it != end; ++it)
    {
        p_teca_variant_array va = ::get_variant_array(it) ?
             ::get_variant_array(it)->new_copy(alloc) : nullptr;

        this->internals->emplace(std::make_pair(::get_file_id(it),
            std::make_pair(va, ::get_const_metadata(it))));
    }
}

// --------------------------------------------------------------------------
void teca_cf_time_axis_data::shallow_copy(const p_teca_dataset &o)
{
    const_p_teca_cf_time_axis_data other =
        std::dynamic_pointer_cast<const teca_cf_time_axis_data>(o);

    if (!other)
        throw teca_bad_cast(safe_class_name(o), "teca_cf_time_axis_data");

    if (this == other.get())
        return;

    this->internals->clear();

    internals_t::const_iterator it = other->internals->begin();
    internals_t::const_iterator end = other->internals->end();
    for (; it != end; ++it)
    {
        unsigned long id = ::get_file_id(it);
        this->internals->emplace(id, it->second);
    }
}

// --------------------------------------------------------------------------
void teca_cf_time_axis_data::swap(const p_teca_dataset &o)
{
    p_teca_cf_time_axis_data other =
        std::dynamic_pointer_cast<teca_cf_time_axis_data>(o);

    if (!other)
        throw teca_bad_cast(safe_class_name(o), "teca_cf_time_axis_data");

    if (this == other.get())
        return;

    internals_t *tmp = this->internals;
    this->internals = other->internals;
    other->internals = tmp;
}

