#ifndef teca_csv_writer_h
#define teca_csv_writer_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_table_fwd.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_csv_writer)

/// teca_csv_writer - writes datasets in CSV format.
/**
an algorithm that writes tabular data in a CSV (comma separated value)
format that is easily ingested by most spreadsheet apps. Each page of
a workbook is written to a file.
*/
class teca_csv_writer : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_csv_writer)
    ~teca_csv_writer();

    // set the output filename. for time series the substring
    // %t% is replaced with the current time step. the substring
    // %e% is replaced with .bin in binary mode and .csv otherwise
    // %s% is replaced with the table name (workbooks only).
    TECA_ALGORITHM_PROPERTY(std::string, file_name)

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // enable binary mode. default off. when not in
    // binary mode a csv format is used.
    TECA_ALGORITHM_PROPERTY(bool, binary_mode)

protected:
    teca_csv_writer();

    int write_table(const std::string &file_name,
        const const_p_teca_table &table);

    int write_csv(const_p_teca_table table, const std::string &file_name);
    int write_bin(const_p_teca_table table, const std::string &file_name);

private:

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string file_name;
    bool binary_mode;
};

#endif
