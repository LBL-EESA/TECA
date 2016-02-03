#include "teca_workbook.h"
#include "teca_table_collection.h"
#include <sstream>

// --------------------------------------------------------------------------
teca_workbook::teca_workbook()
{
    this->tables = teca_table_collection::New();
}

// --------------------------------------------------------------------------
teca_workbook::~teca_workbook()
{}

// --------------------------------------------------------------------------
void teca_workbook::declare_tables(unsigned int n)
{
    for (unsigned int i = 0; i < n; ++i)
    {
       std::ostringstream oss;
       oss << "table_" << i;
       this->declare_table(oss.str());
    }
}

// --------------------------------------------------------------------------
bool teca_workbook::empty() const noexcept
{
    return !this->tables || !this->tables->size();
}

// --------------------------------------------------------------------------
void teca_workbook::copy(const const_p_teca_dataset &o)
{
    const_p_teca_workbook other
        = std::dynamic_pointer_cast<const teca_workbook>(o);

    if (!other)
    {
        TECA_ERROR("Copy failed. Source must be a workbook")
        return;
    }

    this->tables->copy(other->tables);
}

// --------------------------------------------------------------------------
void teca_workbook::shallow_copy(const p_teca_dataset &o)
{
    p_teca_workbook other
        = std::dynamic_pointer_cast<teca_workbook>(o);

    if (!other)
    {
        TECA_ERROR("Copy failed. Source must be a workbook")
        return;
    }

    this->tables->shallow_copy(other->tables);
}

// --------------------------------------------------------------------------
void teca_workbook::copy_metadata(const const_p_teca_dataset &o)
{
    const_p_teca_workbook other
        = std::dynamic_pointer_cast<const teca_workbook>(o);

    if (!other)
    {
        TECA_ERROR("Copy failed. Source must be a workbook")
        return;
    }

    unsigned int n = other->tables->size();
    for (unsigned int i = 0; i < n; ++i)
    {
        p_teca_table table = teca_table::New();
        table->copy_metadata(other->tables->get(i));
        this->tables->append(other->tables->get_name(i), table);
    }
}

// --------------------------------------------------------------------------
void teca_workbook::swap(p_teca_dataset &o)
{
    p_teca_workbook other
        = std::dynamic_pointer_cast<teca_workbook>(o);

    if (!other)
    {
        TECA_ERROR("Copy failed. Source must be a workbook")
        return;
    }

    p_teca_table_collection tmp = this->tables;
    this->tables = other->tables;
    other->tables = tmp;
}

// --------------------------------------------------------------------------
void teca_workbook::to_stream(teca_binary_stream &s) const
{
    this->tables->to_stream(s);
}

// --------------------------------------------------------------------------
void teca_workbook::from_stream(teca_binary_stream &s)
{
    this->tables->from_stream(s);
}

// --------------------------------------------------------------------------
void teca_workbook::to_stream(std::ostream &s) const
{
    this->tables->to_stream(s);
}
