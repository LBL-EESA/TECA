#include "teca_table.h"

#include "teca_variant_array_impl.h"
#include "teca_variant_array_util.h"
#include "teca_binary_stream.h"
#include "teca_dataset_util.h"
#include "teca_string_util.h"
#include "teca_bad_cast.h"

using namespace teca_variant_array_util;


teca_table::impl_t::impl_t() :
    columns(teca_array_collection::New()), active_column(0)
{}


// --------------------------------------------------------------------------
teca_table::teca_table() : m_impl(std::make_shared<teca_table::impl_t>())
{}

// --------------------------------------------------------------------------
void teca_table::clear()
{
    this->get_metadata().clear();
    m_impl->columns->clear();
    m_impl->active_column = 0;
}

// --------------------------------------------------------------------------
bool teca_table::empty() const noexcept
{
    return m_impl->columns->size() == 0;
}

// --------------------------------------------------------------------------
unsigned int teca_table::get_number_of_columns() const noexcept
{
    return m_impl->columns->size();
}

// --------------------------------------------------------------------------
unsigned long teca_table::get_number_of_rows() const noexcept
{
    if (m_impl->columns->size())
        return m_impl->columns->get(0)->size();

    return 0;
}

// --------------------------------------------------------------------------
p_teca_variant_array teca_table::get_column(const std::string &col_name)
{
    return m_impl->columns->get(col_name);
}

// --------------------------------------------------------------------------
const_p_teca_variant_array teca_table::get_column(const std::string &col_name) const
{
    return m_impl->columns->get(col_name);
}

// --------------------------------------------------------------------------
void teca_table::resize(unsigned long n)
{
    unsigned int n_cols = m_impl->columns->size();
    for (unsigned int i = 0; i < n_cols; ++i)
        m_impl->columns->get(i)->resize(n);
}

// --------------------------------------------------------------------------
void teca_table::reserve(unsigned long n)
{
    unsigned int n_cols = m_impl->columns->size();
    for (unsigned int i = 0; i < n_cols; ++i)
        m_impl->columns->get(i)->reserve(n);
}

// --------------------------------------------------------------------------
int teca_table::get_type_code() const
{
    return teca_dataset_tt<teca_table>::type_code;
}

// --------------------------------------------------------------------------
int teca_table::to_stream(teca_binary_stream &s) const
{
    if (this->teca_dataset::to_stream(s)
        || m_impl->columns->to_stream(s))
        return -1;
    return 0;
}

// --------------------------------------------------------------------------
int teca_table::from_stream(teca_binary_stream &s)
{
    this->clear();

    if (this->teca_dataset::from_stream(s)
        || m_impl->columns->from_stream(s))
        return -1;

    return 0;
}

// --------------------------------------------------------------------------
int teca_table::to_stream(std::ostream &s) const
{
    // because this is used for general purpose I/O
    // we don't let the base class insert anything.

    // write the identifier
    s << "# teca_table v1 " << std::endl
        << "# " << TECA_VERSION_DESCR << std::endl;

    // write the calendar and units
    const teca_metadata &md = this->get_metadata();

    if (md.has("calendar"))
    {
        std::string calendar;
        md.get("calendar", calendar);

        std::string units;
        md.get("time_units", units);

        s << "# calendar = \"" << calendar << "\"" << std::endl
            << "# time_units = \"" << units << "\"" << std::endl;
    }

    // first row contains column names. the name is followed by (int)
    // the int tells the array type. this is needed for deserialization
    unsigned int n_cols = m_impl->columns->size();
    if (n_cols)
    {
        std::string col_name;
        const_p_teca_variant_array col;
        unsigned int col_type;

        col = m_impl->columns->get(0);
        col_name = m_impl->columns->get_name(0);
        col_type = col->type_code();

        if (col_name.find_first_of("()") != std::string::npos)
        {
            TECA_WARNING("Writing incompatible table to stream. This data "
                "will not be readable by TECA because parentheses were "
                "used in the column 0 name \"" << col_name << "\"")
        }

        s << "\"" << col_name << "(" << col_type << ")\"";

        for (unsigned int i = 1; i < n_cols; ++i)
        {
            col = m_impl->columns->get(i);
            col_name = m_impl->columns->get_name(i);
            col_type = col->type_code();

            if (col_name.find_first_of("()") != std::string::npos)
            {
                TECA_WARNING("Writing incompatible table to stream. This data "
                    "will not be readable by TECA because parentheses were "
                    "used in the column " << i << " name \"" << col_name << "\"")
            }

            s << ", \"" << col_name << "(" << col_type << ")\"";
        }

        s << std::endl;
    }

    // set the precision such that we get the same floating point
    // value back when deserializzing
    s.precision(std::numeric_limits<long double>::digits10 + 1);
    s.setf(std::ios_base::scientific, std::ios_base::floatfield);

    // the remainder of the data is sent row by row.
    unsigned long long n_rows = this->get_number_of_rows();
    for (unsigned long long j = 0; j < n_rows; ++j)
    {
        if (n_cols)
        {
            VARIANT_ARRAY_DISPATCH(
                m_impl->columns->get(0).get(),
                TT *a = dynamic_cast<TT*>(m_impl->columns->get(0).get());
                NT v = NT();
                a->get(j, v);
                s << v;
                )
            else VARIANT_ARRAY_DISPATCH_CASE(
                std::string,
                m_impl->columns->get(0).get(),
                TT *a = dynamic_cast<TT*>(m_impl->columns->get(0).get());
                NT v = NT();
                a->get(j, v);
                s << "\"" << v << "\"";
                )
            for (unsigned int i = 1; i < n_cols; ++i)
            {
                VARIANT_ARRAY_DISPATCH(
                    m_impl->columns->get(i).get(),
                    TT *a = dynamic_cast<TT*>(m_impl->columns->get(i).get());
                    NT v = NT();
                    a->get(j, v);
                    s << ", " << v;
                    )
                else VARIANT_ARRAY_DISPATCH_CASE(
                    std::string,
                    m_impl->columns->get(i).get(),
                    TT *a = dynamic_cast<TT*>(m_impl->columns->get(i).get());
                    NT v = NT();
                    a->get(j, v);
                    s << ", \"" << v << "\"";
                    )
            }
        }
        s << std::endl;
    }

    return 0;
}

// --------------------------------------------------------------------------
int teca_table::from_stream(std::istream &s)
{
    m_impl->columns->clear();

    // read the stream into a working buffer
    std::streamoff cur_pos = s.tellg();
    s.seekg(0, std::ios_base::end);
    std::streamoff end_pos = s.tellg();
    size_t n_bytes = end_pos - cur_pos;
    s.seekg(cur_pos);

    char *buf = (char*)malloc(n_bytes+1);
    buf[n_bytes] = '\0';

    if (!s.read(buf, n_bytes))
    {
        free(buf);
        TECA_ERROR("Failed to read from the stream")
        return -1;
    }

    // split into lines, and work line by line
    std::vector<char*> lines;
    if (teca_string_util::tokenize(buf, '\n', lines))
    {
        free(buf);
        TECA_ERROR("Failed to split lines")
        return -1;
    }

    size_t n_lines = lines.size();

    // process comment lines, the begin with the # char
    // these may contain metadata such as calendaring info
    size_t lno = 0;
    while ((lno < n_lines) &&
        teca_string_util::is_comment(lines[lno]))
    {
        const char *lp = lines[lno];
        // calendar
        if (strstr(lp, "calendar"))
        {
            std::string calendar;
            if (teca_string_util::extract_string(lp, calendar))
            {
                TECA_ERROR("Invalid calendar (" << lp << ")")
            }
            else
            {
                this->set_calendar(calendar);
            }
        }
        // time units
        if (strstr(lp, "time_units"))
        {
            std::string time_units;
            if (teca_string_util::extract_string(lp, time_units))
            {
                TECA_ERROR("Invalid time_units spec (" << lp << ")")
            }
            else
            {
                this->set_time_units(time_units);
            }
        }
        ++lno;
    }

    // split the header
    std::vector<char *> header;
    if (teca_string_util::tokenize(lines[lno], ',', header))
    {
        free(buf);
        TECA_ERROR("Failed to split fields")
        return -1;
    }
    ++lno;

    // extract the column names and types from each header field
    size_t n_rows = n_lines - lno;
    size_t n_cols = header.size();
    std::vector<std::string> col_names(n_cols);
    std::vector<p_teca_variant_array> cols(n_cols);
    for (size_t i = 0; i < n_cols; ++i)
    {
        int n_match = 0;
        char name[129];
        int code = 0;
        if ((n_match = sscanf(header[i], " \"%128[^(](%d)\"", name, &code)) != 2)
        {
            free(buf);
            TECA_ERROR("Failed to parse column name and type. " << n_match
                << " matches. Line " << lno - 1 << " column " << i << " field \""
                << header[i] << "\"")
            return -1;
        }

        p_teca_variant_array col = teca_variant_array_factory::New(code);
        if (!col)
        {
            free(buf);
            TECA_ERROR("Failed to construct an array for column " << i)
            return -1;
        }

        col->resize(n_rows);

        col_names[i] = name;
        cols[i] = col;
    }

    // allocate a 2D buffer to hold pointers to each cell in the table.
    n_bytes = n_lines*n_cols*sizeof(char*);
    char **data = (char **)malloc(n_bytes);
    memset(data, 0, n_bytes);

    // copy the data from the line buffer into the 2D structure
    for (size_t i = 0; i < n_rows; ++i)
    {
        size_t j = i + lno;
        size_t ii = i*n_cols;

        if (teca_string_util::tokenize(lines[j], ',', n_cols, data + ii))
        {
            free(buf);
            free(data);
            TECA_ERROR("Failed to tokenize row data at row " << j)
            return -1;
        }
    }

    // work column by column
    for (size_t j = 0; j < n_cols; ++j)
    {
        // deserialize the column
        p_teca_variant_array col = cols[j];
        VARIANT_ARRAY_DISPATCH(
            col.get(),
            auto [p_col] = teca_variant_array_util::data<TT>(col);
            const char *fmt = teca_string_util::scanf_tt<NT>::format();
            for (size_t i = 0; i < n_rows; ++i)
            {
                const char *cell = data[i*n_cols + j];
                if (sscanf(cell, fmt, p_col + i) != 1)
                {
                    free(buf);
                    free(data);
                    TECA_ERROR("Failed to convert numeric cell " << i << ", " << j
                        << " \"" << cell << "\" using format \"" << fmt << "\"")
                    return -1;
                }

            }
            )
        else VARIANT_ARRAY_DISPATCH_CASE(
            std::string, col.get(),
            auto [p_col] = teca_variant_array_util::data<TT>(col);
            for (size_t i = 0; i < n_rows; ++i)
            {
                const char *cell = data[i*n_cols + j];
                if (teca_string_util::extract_string(cell, p_col[i]))
                {
                    free(buf);
                    free(data);
                    TECA_ERROR("Failed to convert string cell " << i << ", " << j
                        << " \"" << cell << "\"")
                    return -1;
                }
            }
            )
        else
        {
            TECA_ERROR("Failed to deserialize column " << j << " of type "
                << col->get_class_name())
            return -1;
        }

        // save it
        m_impl->columns->append(col_names[j], col);
    }

    free(buf);
    free(data);

    return 0;
}

// --------------------------------------------------------------------------
void teca_table::copy(const const_p_teca_dataset &dataset, allocator alloc)
{
    const_p_teca_table other
        = std::dynamic_pointer_cast<const teca_table>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_table");

    if (this == other.get())
        return;

    this->clear();

    this->teca_dataset::copy(dataset, alloc);
    m_impl->columns->copy(other->m_impl->columns, alloc);
}

// --------------------------------------------------------------------------
void teca_table::copy(const const_p_teca_table &other,
    unsigned long first_row, unsigned long last_row, allocator alloc)
{
    if (this == other.get())
        return;

    this->clear();

    if (!other)
        return;

    this->teca_dataset::copy(other, alloc);

    size_t n_rows = last_row - first_row + 1;

    unsigned int n_cols = other->get_number_of_columns();
    for (unsigned int i = 0; i < n_cols; ++i)
    {
        auto ocol_name = other->m_impl->columns->get_name(i);
        auto ocol = other->m_impl->columns->get(i);

        p_teca_variant_array ocol_cpy = ocol->new_instance(alloc);
        ocol_cpy->assign(ocol, first_row, n_rows);

        m_impl->columns->append(ocol_name, ocol_cpy);
    }
}

// --------------------------------------------------------------------------
void teca_table::shallow_copy(const p_teca_dataset &dataset)
{
    const_p_teca_table other
        = std::dynamic_pointer_cast<const teca_table>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_table");

    this->clear();

    this->teca_dataset::shallow_copy(dataset);
    m_impl->columns->shallow_copy(other->m_impl->columns);
}

// --------------------------------------------------------------------------
void teca_table::copy_structure(const const_p_teca_table &other)
{
    unsigned int n_cols = other->get_number_of_columns();
    for (unsigned int i = 0; i < n_cols; ++i)
    {
        m_impl->columns->append(other->m_impl->columns->get_name(i),
            other->m_impl->columns->get(i)->new_instance());
    }
}

// --------------------------------------------------------------------------
void teca_table::swap(const p_teca_dataset &dataset)
{
    p_teca_table other
        = std::dynamic_pointer_cast<teca_table>(dataset);

    if (!other)
        throw teca_bad_cast(safe_class_name(dataset), "teca_table");

    this->teca_dataset::swap(dataset);

    std::shared_ptr<teca_table::impl_t> tmp = m_impl;
    m_impl = other->m_impl;
    other->m_impl = tmp;

    m_impl->active_column = 0;
}

// --------------------------------------------------------------------------
void teca_table::concatenate_rows(const const_p_teca_table &other)
{
    if (!other)
        return;

    size_t n_cols = 0;
    if ((n_cols = other->get_number_of_columns()) != this->get_number_of_columns())
    {
        TECA_ERROR("append failed. Number of columns don't match")
        return;
    }

    for (size_t i = 0; i < n_cols; ++i)
        this->get_column(i)->append(other->get_column(i));
}

// --------------------------------------------------------------------------
void teca_table::concatenate_cols(const const_p_teca_table &other, bool deep)
{
    if (!other)
        return;

    unsigned int n_cols = other->get_number_of_columns();
    for (unsigned int i=0; i<n_cols; ++i)
    {
        auto ocol_name = other->m_impl->columns->get_name(i);
        auto ocol = other->m_impl->columns->get(i);

        m_impl->columns->append(ocol_name, deep ? ocol->new_copy() :
            std::const_pointer_cast<teca_variant_array>(ocol));
    }
}
