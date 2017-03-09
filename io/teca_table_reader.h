#ifndef teca_table_reader_h
#define teca_table_reader_h

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
A reader for data stored in binary table format. By default
the reader reads and returns the entire table on rank 0.
The reader can partition the data accross an "index column".
The index column assigns a unique id to rows that should be
returned together. The reader reports the number of unique
ids to the pipeline which can then be requested by the pipeline
during parallel or sequential execution.

output:
    generates a table containing the data read from the file.
*/
class teca_table_reader : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_reader)
    ~teca_table_reader();

    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_table_reader)

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

protected:
    teca_table_reader();

private:
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
    std::vector<std::string> metadata_column_names;
    std::vector<std::string> metadata_column_keys;

    struct teca_table_reader_internals;
    teca_table_reader_internals *internals;
};

#endif
