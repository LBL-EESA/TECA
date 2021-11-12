#ifndef teca_cf_time_axis_reader_h
#define teca_cf_time_axis_reader_h

#include "teca_config.h"
#include "teca_shared_object.h"
#include "teca_algorithm.h"
#include "teca_metadata.h"

#include <string>
#include <vector>
#include <iostream>

TECA_SHARED_OBJECT_FORWARD_DECL(teca_cf_time_axis_reader)

/// An algorithm to read time axis and its attributes in parallel.
class TECA_EXPORT teca_cf_time_axis_reader : public teca_algorithm
{
public:
    TECA_ALGORITHM_STATIC_NEW(teca_cf_time_axis_reader)
    TECA_ALGORITHM_DELETE_COPY_ASSIGN(teca_cf_time_axis_reader)
    TECA_ALGORITHM_CLASS_NAME(teca_cf_time_axis_reader)
    ~teca_cf_time_axis_reader() = default;

    // describe the set of files comprising the dataset. This
    // should contain the full path and regex describing the
    // file name pattern
    TECA_ALGORITHM_PROPERTY(std::string, files_regex)

    // list of file names to open. if this is set the files_regex
    // is ignored.
    TECA_ALGORITHM_VECTOR_PROPERTY(std::string, file_name)

    // set the name of the time axis (time)
    TECA_ALGORITHM_PROPERTY(std::string, t_axis_variable)

    // get the path and files found. only rank 0 will have these
    // these will be populated after the report phase
    const std::string &get_path() const { return this->path; }
    const std::vector<std::string> &get_files() const { return this->files; }

protected:
    teca_cf_time_axis_reader();

    void set_modified() override;

    using teca_algorithm::get_output_metadata;

    teca_metadata get_output_metadata(unsigned int port,
        const std::vector<teca_metadata> &input_md) override;

    const_p_teca_dataset execute(unsigned int port,
        const std::vector<const_p_teca_dataset> &input_data,
        const teca_metadata &request) override;

private:
    std::string t_axis_variable;
    std::string files_regex;
    std::vector<std::string> file_names;
    std::vector<std::string> files;
    std::string path;
};

#endif
