#ifndef teca_table_writer_h
#define teca_table_writer_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <vector>
#include <string>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_table_writer)

/** @brief
 * An algorithm that writes tabular data in a binary or CSV (comma separated
 * value) format that is easily ingested by most spreadsheet apps. Each page
 * of a database is written to a file.
 *
 * @details
 * The binary format is internal to TECA, and provides the best performance.
 *
 * The CSV format is intended for use getting data into other tools such as MS
 * Excel and or Python based codes.
 *
 * See TECA CSV format specification in teca_table_reader for more
 * information.
 */
class TECA_EXPORT teca_table_writer : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_table_writer)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_table_writer)
    TECA_ALGORITHM_CLASS_NAME(teca_table_writer)
    ~teca_table_writer();

    // set the output filename. for time series the substring
    // %t% is replaced with the current time step. the substring
    // %e% is replaced with .bin in binary mode, .csv in csv mode, .nc in
    // netcdf mode, and xlsx in MS Excel mode.
    // %s% is replaced with the table name (workbooks only).
    TECA_ALGORITHM_PROPERTY(std::string, file_name)

    // sets the name of the row variable in the netCDF file
    TECA_ALGORITHM_PROPERTY(std::string, row_dim_name)

    // report/initialize to/from Boost program options
    // objects.
    TECA_GET_ALGORITHM_PROPERTIES_DESCRIPTION()
    TECA_SET_ALGORITHM_PROPERTIES()

    // Select the output file format.
    //
    //      0 : auto, 1 : csv, 2 : bin, 3 : xlsx, 4 : netcdf
    //
    // in the auto format mode, the format is selected using the file
    // name extention. the default is auto.
    enum {format_auto, format_csv, format_bin, format_xlsx, format_netcdf};
    TECA_ALGORITHM_PROPERTY(int, output_format)
    void set_output_format_auto(){ this->set_output_format(format_auto); }
    void set_output_format_csv(){ this->set_output_format(format_csv); }
    void set_output_format_bin(){ this->set_output_format(format_bin); }
    void set_output_format_xlsx(){ this->set_output_format(format_xlsx); }
    void set_output_format_netcdf(){ this->set_output_format(format_netcdf); }

protected:
    teca_table_writer();

private:
    using teca_algorithm::get_output_metadata;

    const_p_teca_dataset execute(
        unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

private:
    std::string file_name;
    std::string row_dim_name;
    int output_format;
};

#endif
