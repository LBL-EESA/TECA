#ifndef teca_table_reader_h
#define teca_table_reader_h

#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_shared_object.h"

#include <vector>
#include <string>
#include <mutex>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_reader)

/// a reader for data stored in binary table format
/**
A reader for data stored in binary table format. For now, this reader reads 
an entire table from a file.

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

protected:
    teca_table_reader();

private:
    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string file_name;
};

#endif
