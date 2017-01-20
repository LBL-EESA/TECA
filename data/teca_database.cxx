#include "teca_database.h"
#include "teca_table_collection.h"
#include <sstream>

// --------------------------------------------------------------------------
teca_database::teca_database()
{
    this->tables = teca_table_collection::New();
}

// --------------------------------------------------------------------------
teca_database::~teca_database()
{}

// --------------------------------------------------------------------------
void teca_database::declare_tables(unsigned int n)
{
    for (unsigned int i = 0; i < n; ++i)
    {
       std::ostringstream oss;
       oss << "table_" << i;
       this->declare_table(oss.str());
    }
}

// --------------------------------------------------------------------------
bool teca_database::empty() const noexcept
{
    return !this->tables || !this->tables->size();
}

// --------------------------------------------------------------------------
void teca_database::copy(const const_p_teca_dataset &o)
{
    const_p_teca_database other
        = std::dynamic_pointer_cast<const teca_database>(o);

    if (!other)
    {
        TECA_ERROR("Copy failed. Source must be a database")
        return;
    }

    this->teca_dataset::copy(o);
    this->tables->copy(other->tables);
}

// --------------------------------------------------------------------------
void teca_database::shallow_copy(const p_teca_dataset &o)
{
    p_teca_database other
        = std::dynamic_pointer_cast<teca_database>(o);

    if (!other)
    {
        TECA_ERROR("Copy failed. Source must be a database")
        return;
    }

    this->teca_dataset::shallow_copy(o);
    this->tables->shallow_copy(other->tables);
}

// --------------------------------------------------------------------------
void teca_database::copy_metadata(const const_p_teca_dataset &o)
{
    const_p_teca_database other
        = std::dynamic_pointer_cast<const teca_database>(o);

    if (!other)
    {
        TECA_ERROR("Copy failed. Source must be a database")
        return;
    }

    this->teca_dataset::copy_metadata(o);

    unsigned int n = other->tables->size();
    for (unsigned int i = 0; i < n; ++i)
    {
        p_teca_table table = teca_table::New();
        table->copy_metadata(other->tables->get(i));
        this->tables->append(other->tables->get_name(i), table);
    }
}

// --------------------------------------------------------------------------
void teca_database::swap(p_teca_dataset &o)
{
    p_teca_database other
        = std::dynamic_pointer_cast<teca_database>(o);

    if (!other)
    {
        TECA_ERROR("Copy failed. Source must be a database")
        return;
    }

    this->teca_dataset::swap(o);

    p_teca_table_collection tmp = this->tables;
    this->tables = other->tables;
    other->tables = tmp;
}

// --------------------------------------------------------------------------
void teca_database::to_stream(teca_binary_stream &s) const
{
    this->teca_dataset::to_stream(s);
    this->tables->to_stream(s);
}

// --------------------------------------------------------------------------
void teca_database::from_stream(teca_binary_stream &s)
{
    this->teca_dataset::from_stream(s);
    this->tables->from_stream(s);
}

// --------------------------------------------------------------------------
void teca_database::to_stream(std::ostream &s) const
{
    this->teca_dataset::to_stream(s);
    this->tables->to_stream(s);
}
