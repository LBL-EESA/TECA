#ifndef teca_cf_reader_h
#define teca_cf_reader_h

#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <vector>
#include <string>
#include <memory>

class teca_cf_reader;
using p_teca_cf_reader = std::shared_ptr<teca_cf_reader>;
using const_p_teca_cf_reader = std::shared_ptr<const teca_cf_reader>;

/// a reader for data stroed in netcdf CF format
/**
metadata keys:

request keys:
*/
class teca_cf_reader : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cf_reader)
    virtual ~teca_cf_reader() = default;

    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cf_reader)

    // describe the set of files comprising the dataset. This
    // should contain the full path and regex describing the
    // file name pattern
    TECA_ALGORITHM_PROPERTY(std::string, files_regex)

    // set the variable to use for the coordinate axes.
    // the defaults are: x => lon, y => lat, z = "",
    // t => "time". leaving z empty will result in a 2D
    // mesh.
    TECA_ALGORITHM_PROPERTY(std::string, x_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, y_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, z_axis_variable)
    TECA_ALGORITHM_PROPERTY(std::string, t_axis_variable)

protected:
    teca_cf_reader();

private:
    virtual
    teca_metadata get_output_metadata(
        unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    virtual
    p_teca_dataset execute(
        unsigned int port,
        const std::vector<p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string files_regex;
    std::string x_axis_variable;
    std::string y_axis_variable;
    std::string z_axis_variable;
    std::string t_axis_variable;
    teca_metadata md;
};

#endif
