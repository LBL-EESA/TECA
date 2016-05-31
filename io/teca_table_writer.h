#ifndef teca_table_writer_h
#define teca_table_writer_h

#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"
#include "teca_table_fwd.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_writer)

/// teca_table_writer - writes tabular datasets in CSV format.
/**
an algorithm that writes tabular data in a CSV (comma separated value)
format that is easily ingested by most spreadsheet apps. Each page of
a database is written to a file.
*/
class teca_table_writer : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_writer)
    ~teca_table_writer();

    // set the output filename. for time series the substring
    // %t% is replaced with the current time step. the substring
    // %e% is replaced with .bin in binary mode and .csv otherwise
    // %s% is replaced with the table name (workbooks only).
    TECA_ALGORITHM_PROPERTY(std::string, file_name)

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // Select the output file format. 0 : csv, 1 : bin, 2 : xlsx.
    // the default is csv.
    enum {format_csv, format_bin, format_xlsx, format_auto};
    TECA_ALGORITHM_PROPERTY(int, output_format)
    void set_output_format_csv(){ this->set_output_format(format_csv); }
    void set_output_format_bin(){ this->set_output_format(format_bin); }
    void set_output_format_xlsx(){ this->set_output_format(format_xlsx); }
    void set_output_format_auto(){ this->set_output_format(format_auto); }

protected:
    teca_table_writer();

private:
    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string file_name;
    int output_format;
};

#endif
