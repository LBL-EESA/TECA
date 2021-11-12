#ifndef teca_table_reader_h
#define teca_table_reader_h

#include "teca_config.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"
#include "teca_table.h"

#include <vector>
#include <string>
#include <mutex>


TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_reader)

/// a reader for data stored in binary table format
/**
 * A reader for data stored in CSV or binary table format. By default the reader
 * reads and returns the entire table on rank 0.  The reader can partition the
 * data across an "index column".  The index column assigns a unique id to rows
 * that should be returned together. The reader reports the number of unique ids
 * to the pipeline which can then be requested by the pipeline during parallel or
 * sequential execution.
 *
 * output:
 *     generates a table containing the data read from the file.
 *
 *
 * ### TECA CSV format specification
 *
 * #### Comment lines
 *
 * a '#' character at the start of a line marks it as a comment. The version of
 * the CSV specification as well as the version of TECA used to write the table
 * will be stored in comment lines. Comment lines are currently skipped when
 * reading the table.
 *
 * #### Column definitions
 *
 * the first row stores the names and data types of the columns. Column names are
 * strings and delimited by double quotes. A column's data type is
 * encoded in the name using (N) where N is an integer type code defined by
 * teca_variant_array and parentheses delimit the type code. The type code
 * sequence is stripped from the name when the file is read.
 *
 * | C type             | code |
 * | ------             | ---- |
 * | char               | 1    |
 * | unsigned char      | 2    |
 * | int                | 3    |
 * | unsigned int       | 4    |
 * | short int          | 5    |
 * | short unsigned int | 6    |
 * | long               | 7    |
 * | unsigned long      | 8    |
 * | long long          | 9    |
 * | unsigned long long | 10   |
 * | float              | 11   |
 * | double             | 12   |
 * | std::string        | 13   |
 *
 * The number of column definitions found determines the number of columns in the
 * table when reading.
 *
 * #### Column data
 *
 * Data is organized row by row with an entry for each column. Entries are
 * separated by commas ','. Error's will occur when the number of column
 * definitions don't match the number of data entries per row.
 *
 * #### String data
 *
 * Strings are delimited by double quotations. Double quotes and commas in strings
 * may be escaped by a backslash.
 *
 * #### Numeric data
 *
 * The type code provided in the column definition tells the type of number.
 * These codes are defined in teca_variant_array.
 *
 * floating point types are written with format and precision such that they
 * may be read without introducing rounding error.
 */
class TECA_EXPORT teca_table_reader : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_reader)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_table_reader)
    TECA_ALGORITHM_CLASS_NAME(teca_table_reader)
    ~teca_table_reader();

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // the file from which data will be read.
    TECA_ALGORITHM_PROPERTY(std::string, file_name)

    // name of the column containing index values.
    // if this is not empty the reader will operate
    // in parallel mode serving up requested indices
    // on demand. otherwise rank 0 reads the entire
    // table regardless of what is requested.
    TECA_ALGORITHM_PROPERTY(std::string, index_column)

    // when set a column named "original_ids" is placed
    // into the output. values map back to the row number
    // of the source dataset. By default this is off.
    TECA_ALGORITHM_PROPERTY(int, generate_original_ids)

    // name of columns to copy directly into metadata
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, metadata_column_name)

    // keys that identify metadata columns
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, metadata_column_key)

    // add a metadata column with the given key
    void add_metadata_column(const std::string &column, const std::string &key)
    {
        this->append_metadata_column_name(column);
        this->append_metadata_column_key(key);
    }

    // removes all metadata columns
    void clear_metadata_columns()
    {
        this->clear_metadata_column_names();
        this->clear_metadata_column_keys();
    }

    // Select the output file format. 0 : csv, 1 : bin, 2 : xlsx, 3 : auto
    // the default is csv.
    enum {format_csv, format_bin, format_xlsx, format_auto};
    TECA_ALGORITHM_PROPERTY(int, file_format)
    void set_file_format_csv(){ this->set_file_format(format_csv); }
    void set_file_format_bin(){ this->set_file_format(format_bin); }
    void set_file_format_xlsx(){ this->set_file_format(format_xlsx); }
    void set_file_format_auto(){ this->set_file_format(format_auto); }

protected:
    teca_table_reader();

private:
    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    void set_modified() override;
    void clear_cached_metadata();

private:
    std::string file_name;
    std::string index_column;
    int generate_original_ids;
    int file_format;
    std::vector<std::string> metadata_column_names;
    std::vector<std::string> metadata_column_keys;

    struct teca_table_reader_internals;
    teca_table_reader_internals *internals;
};

#endif
